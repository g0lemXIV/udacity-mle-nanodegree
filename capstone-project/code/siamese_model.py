"""Siamese network model based on https://keras.io/examples/vision/siamese_network/
article with distance metric layer implementation.
"""
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, data_shape, n_components, image_shape=(62, 47), margin=1.0):
        super(SiameseModel, self).__init__()
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.image_shape = image_shape
        self.data_shape = data_shape
        self.n_components = n_components
        # build simaese network
        self.embedding = self.embedding_network_builder()
        self.siamese_network_builder()

    def call(self, inputs):
        return self.siamese_network(inputs)

    def resnet_network_builder(self):
        """
        This builder is responsible for creating the ResNet50 Layer which is used to
        embed image data into a lower dimension. It also unfreezes the last convolutional
        layers for retraining.
        """
        base_cnn = resnet.ResNet50(
            weights="imagenet", input_shape=self.image_shape + (3,), include_top=False
        )
        # unfreeze layers
        for layer in base_cnn.layers:
            if "conv5" in layer.name or "conv4" in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False
        return base_cnn

    def siamese_network_builder(self):
        """
        This builder is responsible for build the siamese model. It builds inputs layer
        for each sample, then embed it with embedding model and final measure
        distance between pairs with DistanceLayer
        """
        anchor_input = layers.Input(name="anchor", shape=self.data_shape)
        positive_input = layers.Input(name="positive", shape=self.data_shape)
        negative_input = layers.Input(name="negative", shape=self.data_shape)
        distances = DistanceLayer()(
            self.embedding(anchor_input),
            self.embedding(positive_input),
            self.embedding(negative_input),
        )
        # build siamese model
        self.siamese_network = Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=distances
        )

    def embedding_network_builder(self):
        """
        This builder is responsible for build the embedding model.
        Which is responsible for embedding, also it will be used in the production
        environment to calculate embedding for samples.
        """
        input_layer = tf.keras.Input(shape=(self.data_shape,))
        reshape_image = tf.reshape(input_layer, [-1, self.image_shape[0], self.image_shape[1]])
        stack_image = tf.stack([reshape_image]*3, axis=-1)
        image = tf.image.convert_image_dtype(stack_image, tf.float32)
        resnet_input = resnet.preprocess_input(image)
        base_cnn = self.resnet_network_builder()(resnet_input)

        flatten = layers.Flatten()(base_cnn)
        dense1 = layers.Dense(512, kernel_regularizer='l2')(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Activation('relu')(dense1)
        dense1 = layers.Dropout(0.3)(dense1)
        dense2 = layers.Dense(256,  kernel_regularizer='l2')(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        dense2 = layers.Activation('relu')(dense2)
        dense2 = layers.Dropout(0.3)(dense2)
        output = layers.Dense(self.n_components)(dense2)


        embedding = Model(input_layer, output, name="Embedding")
        return embedding



    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

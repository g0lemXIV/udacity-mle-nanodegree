"""Main script used to train and save tensorflow siamese network model"""

import argparse
import os
import numpy as np
from siamese_model import SiameseModel
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks as clb


def make_callbacks():
    """
    Add callbacks to tensorflow model which monitor val loss
    """
    callbacks = [
        clb.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=25, min_rl=0.00001),
        clb.EarlyStopping(monitor='val_loss', patience=50)
    ]
    return callbacks


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--n_components', type=int, default=256)
    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args, _ = parser.parse_known_args()

    # load datasets
    train_dataset = os.path.join(os.environ.get('SM_CHANNEL_TRAIN'), 'X_train_siamese.npy')
    test_dataset = os.path.join(os.environ.get('SM_CHANNEL_TRAIN'), 'X_test_siamese.npy')
    train_data = np.load(train_dataset)
    test_data = np.load(test_dataset)

    # split dataset to anchor, positive and negative samples of images
    anchor_train, positive_train, negative_train = train_data[:, 0, :], train_data[:, 1, :], train_data[:, 2, :]

    anchor_test, positive_test, negative_test = test_data[:, 0, :], test_data[:, 1, :], test_data[:, 2, :]

    # create Siamese network model and fit with the data
    siamese_model = SiameseModel(data_shape=train_data.shape[-1], n_components=args.n_components)
    siamese_model.compile(optimizer=optimizers.Adam(args.learning_rate))
    siamese_model.fit([anchor_train, positive_train, negative_train],
                      validation_data=([anchor_test, positive_test, negative_test]),
                      callbacks=make_callbacks(),
                      epochs=args.epochs, batch_size=args.batch_size)

    # Save final trained model version
    version = "00000000"
    ckpt_dir = os.path.join(args.model_dir, version)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    siamese_model.embedding.save(ckpt_dir)
# Machine Learning Engineering Capstone Project



![udacity](https://miro.medium.com/max/1200/1*8chqKIGajMlaANgxyPyyXg.jpeg)

#### In this project, I apply what I have learned on Machine Learning Engineer course 


### Project description

---------------------------
The project includes all necessary steps to create an end-to-end machine learning model for face reidentification, the project was divided to 3 steps:
  - Downloading, preprocessing, and analyzing data made in 1_Data_Analysis.ipynb
  - Creating base PCA model in 2_PCA_Model.ipynb  
  - Creating Siamese Network in 3_Siamese_Model.ipynb

**The purpose of this project is to answer the following questions:**  
  - Is it possible to creat reidentification system based on free available datasets?
  - How good in terms of accuracy deep learning model is?
  - How to compare faces in unsupervised style?

### Steps taken in the project  
A brief description of the steps that have been taken in the project
1. Downloading LFW dataset from the scikit-learn library.
2. Analyze the dataset and create a train, test, and validation sets.
3. Train, deploy and examine base PCA model with Sagemaker SDK.
4. Train, deploy and examine Siamese Network model with Sagemaker in script mode.
5. Examine results.
------------------------------   

### The choice of tools, data and libraries  

#### Tools

-----------------------------------------------------
- AWS S3 - Amazon Simple Storage Service (Amazon S3) is an object storage service that offers industry-leading scalability, data availability, security, and performance.  
- Amazon Sagemaker - Amazon SageMaker helps data scientists and developers to prepare, build, train, and deploy high-quality machine learning (ML) models quickly by bringing together a broad set of capabilities purpose-built for ML.

#### Data

-----------------------------------------------------
For training and testing purpose and for input/output data as well I will use LFW Face Database. Labeled Faces in the Wild is a public benchmark for face verification, also known as pair matching.  

<img src="https://storage.googleapis.com/tfds-data/visualization/fig/lfw-0.1.0.png" height="250" width="250" class="center">

#### Libraries

-----------------------------------------------------
 - Tensorflow 2.0+
 - Scikit-learn
 - Numpy
 - Pandas
 - Seaborn
 - Sagemaker
 - Boto3
  
### Results

------------------------------------
Results and larger describe can be seen in notebooks and project report file

### Next steps and improvement 

------------------------------------

The project involves the development of a model to verify a person using computer vision algorithms. The future project steps could involve the development of an application to verify a person using computer vision algorithms. The first step to improve would be to collect more data and perform new training with additional hyperparameter tuning and augmentation process, at this stage, it is also assumed to add data from other sets in order to normalize the set in terms of ethnic and age diversity.  

Next, models to extract faces from the image based on simple computer vision algorithms would be created. Also in this step, tracking face with preventing adversarial attacks It is also assumed to create a web application on which the user will be able to place a photo and re-identify it later based on registration information. For this backend pipeline AWS Lambda, DynamoDB, and SageMaker could be used. 

# Cardiovascular-Disease-Risk-Prediction

## Description
The main purpose of this project is to demonstrate the use of ZenML for training and deploying the ML pipelines in following ways:
* By offering you a framework and template to base your own work on.
* By integrating with tools like MLflow for deployment, tracking and more
* By allowing you to build and deploy your machine learning pipelines easily

The main librares used for this projects are
1. Zenml
2. catboost
3. mlflow

The dataset contains features like Age, Height, Exercise, etc to predict the likelihood of Cardiovascular disease risk.

In this Project, we give special consideration to the MLflow integration of ZenML. In particular, we utilize MLflow tracking to track our metrics and parameters, and MLflow deployment to deploy our model. We also use Streamlit to showcase how this model will be used in a real-world setting.

## Contents

The training pipeline consists of several steps which are:

**Training pipeline**
* ingest_data: This step will ingest the data and create a DataFrame.
* clean_data: This step will clean the data and remove the unwanted columns.
* model_train: This step will train the model and save the model using MLflow autologging.
* evaluation: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.

**Deployment pipeline**
We have another pipeline, the deployment.py, that extends the training pipeline, and implements a continuous deployment workflow. It ingests and processes input data, trains a model and then (re)deploys the prediction server that serves the model if it meets our evaluation criteria.

## Running the pipeline
We can run the training pipeline by running the following code.

* The training pipeline:
  ```bash
  python run_pipeline.py
  ```

* The continuous deployment pipeline:
  ```bash
  python run_deployment.py
  ```

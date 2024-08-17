# Project Description:
This project showcases a data pipeline for predicting customer churn using historical customer data. The pipeline leverages Hadoop for distributed data storage and PySpark for data processing and machine learning. The goal is to identify patterns and factors leading to customer churn, enabling businesses to take proactive measures.

# Key Features:
Data Ingestion: Efficiently loads customer data from local storage into Hadoop Distributed File System (HDFS).

Data Processing: Utilizes PySpark for cleaning, transforming, and feature engineering to prepare data for machine learning.

Model Training: Implements a machine learning model to predict customer churn based on historical data.

Evaluation: Assesses the performance of the model using metrics such as accuracy, precision, recall, and F1-score.

Deployment: Provides a mechanism to deploy the model for real-time predictions.

# Data Processing Steps:
Ingestion: Loads CSV customer data from a specified local directory into HDFS.

Cleaning: Handles missing values, filters out irrelevant data, and standardizes features.

Transformation: Performs feature engineering to create meaningful features for the machine learning model.

Model Training: Uses PySpark MLlib to train a classification model (e.g., logistic regression or random forest) to predict customer churn.

Evaluation: Evaluates the model’s performance with metrics and saves the results.

Deployment: Provides scripts or instructions for deploying the model in a real-time environment.

# Technologies Used:
Hadoop: For distributed storage and management of large datasets.

PySpark: For scalable data processing, feature engineering, and machine learning.

HDFS: For efficient data storage and retrieval.

PySpark MLlib: For building and evaluating machine learning models.



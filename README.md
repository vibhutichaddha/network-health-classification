# Network Health Classification Using Telecom KPI Data

## Overview

This project focuses on analyzing telecom network Key Performance Indicator (KPI) data and building machine learning models to classify network cells into three categories:

* Healthy
* Degraded
* Congested

The project covers the complete machine learning workflow, including data exploration, visualization, feature engineering, preprocessing, model development, hyperparameter tuning, cross-validation, and performance evaluation.

## Dataset

The telecom KPI dataset contains the following attributes:

* Cell_ID
* Timestamp
* RSRP
* SINR
* Latency
* Throughput
* Packet_Loss
* Connected_Users

The target column **Network_Status** is created using predefined KPI thresholds to classify each network cell.

# Task 1: Data Exploration & Understanding

The dataset is explored to understand its structure, quality, and statistical properties.

### Activities

* Loaded the dataset using Pandas
* Checked dataset dimensions
* Examined data types
* Identified missing values
* Detected duplicate records
* Generated statistical summaries

### Deliverables

* Dataset summary report
* Exploratory Data Analysis (EDA)

# Task 2: Data Visualization

Various visualizations are created to understand network behavior and KPI relationships.

### Visualizations

* Network Status Distribution
* SINR vs Throughput Scatter Plot
* RSRP Distribution Histogram
* Correlation Heatmap

### Outcome

The visualizations help identify signal quality patterns, KPI relationships, and the overall distribution of network conditions.

# Task 3: Feature Engineering

New features are created to improve the quality of information available for machine learning models.

### Engineered Features

* **Signal_Quality**

  * `0.6 × SINR + 0.4 × (RSRP + 120)`

* **Network_Load**

  * `Connected_Users / Maximum_Users`

* **Network_Health_Index**

  * `(Throughput / SINR) - Latency - Packet_Loss`

### Outcome

The dataset is enriched with meaningful features representing signal quality, network utilization, and overall network performance.

# Task 4: Data Preprocessing

The dataset is cleaned and prepared for machine learning.

### Activities

* Handled missing values
* Removed duplicate records
* Encoded target labels using LabelEncoder
* Applied feature scaling using StandardScaler
* Split the dataset into:

  * 80% Training Data
  * 20% Testing Data

### Outcome

A clean, normalized, and machine learning-ready dataset is produced for model training.

# Task 5: Machine Learning Model Development

Three supervised machine learning models are trained to classify network health.

### Models Implemented

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### Objective

Train multiple classification algorithms and compare their performance on the telecom KPI dataset.

# Task 6: Cross Validation

To evaluate the stability and reliability of each model, **5-Fold Cross Validation** is performed.

### Metrics Reported

* Mean Accuracy
* Standard Deviation

### Outcome

Cross-validation provides a more reliable estimate of model performance by evaluating each model across multiple train-test splits.

# Task 7: Hyperparameter Tuning

Model performance is optimized using **GridSearchCV**.

### Decision Tree Parameters

* max_depth
* min_samples_split
* criterion

### Random Forest Parameters

* n_estimators
* max_depth
* min_samples_split

### Outcome

The best combination of hyperparameters is selected based on cross-validation accuracy, improving the overall performance of the models.

# Task 8: Model Evaluation

The trained models are evaluated using multiple performance metrics.

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score

### Visualizations

* Confusion Matrix
* ROC Curve (One-vs-Rest Strategy)

### Outcome

The evaluation metrics and visualizations provide a comprehensive comparison of all three models and help identify the best-performing classifier for network health classification.

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

# Author

Vibhuti Chaddha

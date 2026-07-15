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

# Task 9: Feature Importance Analysis

Feature importance analysis is performed using the **Random Forest Classifier** to identify the KPIs that contribute the most to network health prediction.

### Activities

* Calculated feature importance scores using Random Forest.
* Ranked all features based on their contribution.
* Identified the **Top 5 most important KPIs**.
* Visualized feature importance using a bar chart.

### Outcome

The analysis highlights the KPIs that have the greatest influence on network health classification, helping understand which network parameters are most critical for prediction.

# Task 10: Model Comparison

The performance of all trained machine learning models is compared using the evaluation metrics obtained during model testing.

### Models Compared

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### Comparison Metrics

* Accuracy
* Precision
* Recall
* F1 Score

### Outcome

A comparison table is created to evaluate the strengths and weaknesses of each model. Based on the evaluation results, the **Random Forest Classifier** is selected as the best-performing model due to its superior accuracy and overall classification performance.

### Report Sections

1. Problem Statement
2. Dataset Description
3. Features Used
4. Models Evaluated
5. Evaluation Results
6. Best Model Selection
7. Business Recommendations

### Outcome

The report provides a clear overview of the project, explains the methodology, summarizes the experimental results, and presents practical recommendations for improving telecom network monitoring and optimization.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

# Complete Project Workflow

```text
Telecom KPI Dataset
        │
        ▼
Task 1
Data Exploration & Understanding
        │
        ▼
Task 2
Data Visualization
        │
        ▼
Task 3
Feature Engineering
        │
        ▼
Task 4
Data Preprocessing
        │
        ▼
Task 5
Machine Learning Model Development
        │
        ▼
Task 6
5-Fold Cross Validation
        │
        ▼
Task 7
Hyperparameter Tuning (GridSearchCV)
        │
        ▼
Task 8
Model Evaluation
(Accuracy, Precision, Recall, F1 Score,
Confusion Matrix, ROC Curve)
        │
        ▼
Task 9
Feature Importance Analysis
        │
        ▼
Task 10
Model Comparison
        │
        ▼
Task 11
Technical Report
        │
        ▼
Best Model Selection &
Business Recommendations
```

# Conclusion

This project demonstrates a complete end-to-end machine learning pipeline for telecom network health classification. Beginning with raw telecom KPI data, the workflow includes data exploration, visualization, feature engineering, preprocessing, model training, cross-validation, hyperparameter tuning, model evaluation, feature importance analysis, and final model comparison. The Random Forest Classifier achieved the best overall performance and is recommended for deployment to support automated network monitoring, proactive fault detection, and improved service quality in telecom networks.

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

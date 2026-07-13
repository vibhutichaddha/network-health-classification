# Network Health Classification Using Telecom KPI Data

## Overview

This project focuses on preparing telecom network KPI data for machine learning by performing data exploration, visualization, feature engineering, and preprocessing. The objective is to analyze network performance indicators and create a clean, feature-rich dataset that can be used to classify telecom cells as **Healthy**, **Degraded**, or **Congested** in subsequent machine learning tasks.

---

## Dataset

The dataset contains the following telecom Key Performance Indicators (KPIs):

* Cell_ID
* Timestamp
* RSRP
* SINR
* Latency
* Throughput
* Packet_Loss
* Connected_Users

## Task 1: Data Exploration & Understanding

The dataset is explored to understand its structure and quality before applying machine learning techniques.

### Activities Performed

* Loaded the dataset using Pandas.
* Checked the dataset dimensions.
* Examined column names and data types.
* Identified missing values.
* Detected duplicate records.
* Generated statistical summaries of numerical features.

### Outcome

A complete understanding of the dataset, including its quality, structure, and basic statistical properties.

## Task 2: Data Visualization

Visualizations are created to analyze network performance and identify relationships among KPIs.

### Visualizations

* Network Status Distribution
* SINR vs Throughput Scatter Plot
* RSRP Distribution Histogram
* Correlation Heatmap

### Outcome

The visualizations provide insights into network conditions, signal quality, signal strength distribution, and relationships between different KPIs.

## Task 3: Feature Engineering

New features are created from the existing KPIs to improve the quality of information available for machine learning models.

### Engineered Features

* **Signal_Quality**

  * `0.6 × SINR + 0.4 × (RSRP + 120)`

* **Network_Load**

  * `Connected_Users / Maximum_Users`

* **Network_Health_Index**

  * `(Throughput / SINR) - Latency - Packet_Loss`

### Outcome

The dataset is enriched with additional features that better represent overall network performance and utilization.

## Task 4: Data Preprocessing

The dataset is cleaned and prepared for machine learning.

### Activities Performed

* Handled missing values.
* Removed duplicate records.
* Encoded target labels using LabelEncoder.
* Applied feature scaling using StandardScaler.
* Split the dataset into:

  * 80% Training Data
  * 20% Testing Data

### Outcome

A clean, normalized, and machine learning-ready dataset suitable for training and evaluating classification models.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Project Outcome

By completing these four tasks, the telecom KPI dataset is transformed into a clean and feature-engineered dataset with meaningful visual insights and properly preprocessed data. This serves as the foundation for building and evaluating machine learning models for network health classification.

# Author
Vibhuti Chaddha

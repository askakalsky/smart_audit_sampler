# Script Description

This script performs **anomaly-based sampling** using the **Local Outlier Factor (LOF)** method to detect outliers in a dataset. LOF identifies anomalies based on the local density of records by comparing each record with its neighbors.

## How Local Outlier Factor works:

### Basics of the Local Outlier Factor Algorithm:
- **Local Outlier Factor (LOF)** is a method for detecting anomalies that evaluates how much each record differs from its neighbors in a multidimensional space.
- LOF uses local density to identify anomalous records. Records located in less dense regions of the space are considered anomalies.
- **Anomalies** are those records that have low density compared to their nearest neighbors, meaning they are "isolated" in the data space.

### How it works:

### Data Preparation:
- Copies of the original and preprocessed datasets are created to avoid modifying the original data.
- Columns for storing anomaly scores and labels are added to the datasets.

### Training the Local Outlier Factor model:
- The **Local Outlier Factor (LOF)** model is created with default hyperparameters. The algorithm evaluates how each record differs from its neighbors.
- The model is trained on the preprocessed dataset.

### Anomaly scoring:
- After training the model, **anomaly scores** are calculated for each record. Records with the lowest scores are considered the most anomalous.
- Anomaly scores and labels are added to both datasets (original and preprocessed).

### Selecting the sample:
- The most anomalous samples are selected based on their anomaly scores. If the number of detected anomalies exceeds the sample size, random sampling is performed from the anomalies.

### Forming the sample:
- The most anomalous records are added to the final sample, and these records are marked in the `is_sample` column.
- If fewer anomalies are detected than the requested sample size, a warning is issued.

### Execution report:
- A summary report of the sampling process is generated, including the population size, number of detected anomalies, date and time of execution, model hyperparameters, and sample size.

---

## Important points:

- **Local density analysis**: LOF compares the density of each record with its nearest neighbors, allowing it to detect isolated records as anomalies.
- **LOF Model**: The model uses local distances between records to assess anomalies, making it effective for detecting local deviations in the data.
- **Reproducibility**: If a random seed is used, the anomaly selection process can be reproduced with the same results.

---

## Potential deviations:

- **Few anomalies**: If there are few anomalies in the population, the sample may be smaller than the requested size.
- **Hyperparameter tuning**: Using default hyperparameters may not always yield optimal results for specific datasets.

---

## In conclusion:

- **Local Outlier Factor (LOF)** is an effective method for detecting local anomalies by using the density of neighboring records. This script allows you to sample based on these anomalies and form a final set for further analysis.

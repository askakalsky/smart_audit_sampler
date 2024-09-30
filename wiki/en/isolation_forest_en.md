# Script Description

This script performs **anomaly-based sampling** using the **Isolation Forest** method, which is used to detect outliers in a dataset. Isolation Forest uses a tree-based structure to isolate anomalous points, allowing it to quickly and efficiently identify outliers.

## How Isolation Forest works:

### The basics of the Isolation Forest algorithm:
- **Isolation Forest** is an algorithm designed to isolate points in a multidimensional space. The goal of the algorithm is to isolate each point (record) from the rest of the data.
- Each record is isolated by splitting the data into subsets using isolation trees, where anomalous records are isolated faster than normal ones because fewer splits are required.
- **Anomalies** are defined as those records that are isolated quickly, meaning they require fewer splits in the tree-building process.

### How it works:

### Data Preparation:
- Copies of the original and preprocessed datasets are created to avoid modifying the original data.
- Columns for storing anomaly scores and labels are added to the datasets.

### Training the Isolation Forest model:
- An **Isolation Forest** model is created using default hyperparameters and a random seed for reproducibility.
- **Isolation Forest** constructs several isolation trees, where each tree isolates points by randomly splitting features and values.

### Anomaly scoring:
- After training the model, **anomaly scores** are computed for each record. The fewer splits required to isolate a record, the more anomalous it is considered to be.
- Anomaly scores and labels are added to both datasets (original and preprocessed).

### Selecting the sample:
- Both datasets are sorted by anomaly scores, starting from the lowest values.
- The most anomalous samples are selected based on the requested sample size.

### Forming the sample:
- The most anomalous records are added to the final sample, and these records are marked in the `is_sample` column.

### Execution report:
- A summary report of the sampling process is generated, including population size, the number of anomalies detected, the date and time of execution, the model's hyperparameters, and the sample size.

---

## Important points:

- **Isolation mechanism**: Records that are isolated more quickly are considered more anomalous and are selected for further analysis.
- **Isolation Forest model**: This method is based on using isolation trees and detects anomalies based on the number of splits required to isolate a point.
- **Reproducibility**: If a random seed is used, the model and sampling process can be reproduced with the same results.

---

## Potential deviations:

- **Few anomalies**: If the population contains very few anomalies, the sample may include records that are not significantly anomalous.
- **Hyperparameter tuning**: Using default hyperparameters may not provide optimal results for specific datasets.

---

## In conclusion:

- **Isolation Forest** is an effective method for detecting anomalies, using a tree-based structure to isolate records. This script allows you to sample based on these anomalies and form a final set for further analysis.

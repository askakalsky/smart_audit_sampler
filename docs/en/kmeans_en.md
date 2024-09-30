# Script Description

This script performs **sampling based on K-Means clustering** with hyperparameter optimization using **Optuna**. The sampling is based on the distance of records from cluster centroids, where the most "distant" records are considered the most representative of their cluster.

## How the K-Means method works:

### Basics of the K-Means Algorithm:
- **K-Means** is a clustering method that partitions data into **k clusters** based on the similarity between points. The algorithm iteratively updates the positions of the cluster centroids to minimize the distance between the points and their centroids.
- After clustering, each point belongs to the cluster with the nearest centroid.
- **Hyperparameters** (such as the number of clusters, the number of initial iterations, and the maximum number of iterations) can affect the quality of the clustering, so they are optimized using **Optuna**.

### How it works:

### Data Preparation:
- Copies of the original and preprocessed datasets are created to avoid modifying the original data.
- Columns for storing distances to centroids and cluster labels are added to the datasets.

### Hyperparameter optimization for K-Means:
- The **Optuna** library is used to optimize the K-Means hyperparameters, such as the number of clusters, the number of initial iterations, and the maximum number of iterations.
- The optimization is based on the **Calinski-Harabasz** score, which measures the density and separation of the clusters.

### Selecting the sample:
- After optimization, clustering is performed, and the distance to the centroid is calculated for each record.
- To form the sample, records with the largest distances to their centroids are selected, as they are the most representative of their cluster.

### Forming the sample:
- The selected records are added to the final sample, and these records are marked in the `is_sample` column.
- If the requested sample size exceeds the number of available records, a warning is issued.

### Execution report:
- A summary report of the sampling process is generated, including the number of clusters, model parameters, the date and time of execution, the sample size, and the hyperparameters that were optimized.

---

## Important points:

- **Optimization with Optuna**: The K-Means hyperparameters are optimized using Optuna to improve the clustering quality.
- **Distance-based selection**: Records are selected based on their distance from the centroids, allowing the selection of the most representative samples.
- **Reproducibility**: Using a random seed ensures that the clustering and sampling results can be reproduced.

---

## Potential deviations:

- **Small number of clusters**: If the number of clusters is too small, it may result in poor clustering, which will affect the quality of the sample.
- **Large distances may not always be ideal**: Sampling based on the largest distances may not always align with the sampling goals for every use case.

---

## In conclusion:

- This script uses **K-Means clustering** with hyperparameter optimization via **Optuna**. The sampling is based on the distances to cluster centroids, which helps identify the most representative records.

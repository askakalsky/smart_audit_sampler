# Script Description

This script performs **anomaly-based sampling** using the **HDBSCAN** clustering algorithm with hyperparameter optimization via **Optuna**. The algorithm clusters the data, identifies anomalies, and returns a sample of anomalous records.**IMPORTANT** To ensure fast operation of all program modules, this algorithm is limited to use on datasets containing more than 500 000 records.

## How HDBSCAN works:

### Basics of the HDBSCAN Algorithm:
- **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)** is a clustering algorithm that does not require a predefined number of clusters. It groups data based on density and identifies anomalies as those records that do not belong to any cluster.
- HDBSCAN evaluates the density of data at different scales and uses a dendrogram to form clusters. Records that do not fit into any cluster are labeled as anomalies.

### How it works:

### Data Preparation:
- Copies of the original and preprocessed datasets are created to avoid modifying the original data.
- Columns for storing cluster information and anomaly labels are added.

### Hyperparameter optimization with Optuna:
- **Optuna** is used to optimize the hyperparameters of HDBSCAN to maximize the **silhouette score**, which assesses the quality of clustering.
- Parameters such as minimum cluster size and the number of samples are automatically tuned.

### Anomaly detection:
- After optimization, data is clustered, and records that do not belong to any cluster are considered anomalies (labeled as -1).
- Columns for storing cluster information and anomaly labels are added to both datasets.

### Selecting the sample:
- Anomalous records are selected based on the requested sample size. If the number of detected anomalies is smaller than the requested sample size, all anomalies are included in the sample.

### Forming the sample:
- The selected anomalies are added to the final sample, and these records are marked in the `is_sample` column.

### Execution report:
- A summary report of the sampling process is generated, including the number of detected anomalies, model parameters, sample size, and execution date.

---

## Important points:

- **Dynamic clustering with HDBSCAN**: HDBSCAN automatically determines the number of clusters, allowing the discovery of the underlying data structure without the need for a fixed number of clusters.
- **Optuna-based optimization**: HDBSCAN's hyperparameters are automatically tuned to achieve better clustering quality.
- **Reproducibility**: Using a random seed ensures that the clustering and sampling results can be reproduced.

---

## Potential deviations:

- **Few anomalies**: If the number of anomalies in the population is less than the requested sample size, all available anomalies may be returned, which could be fewer than requested.
- **Parameter sensitivity**: The parameters of HDBSCAN can significantly impact the clustering results, making optimization crucial.

---

## In conclusion:

- This script uses **HDBSCAN** for detecting anomalies and clustering data. The **Optuna**-based parameter optimization ensures high-quality clustering, allowing the discovery of data structure and the most anomalous records.

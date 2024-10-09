# Script Description

This script performs **sampling based on K-Means clustering** with hyperparameter optimization using **Optuna**. Sampling is based on the distance of records to the cluster centroids, where the most "distant" records are selected, considered anomalous or best representative for their cluster. It accounts for varying cluster densities to ensure a uniform selection of anomalies from each cluster.

## How the K-Means Method Works

### Basics of the K-Means Algorithm
- **K-Means** is a clustering method that divides data into **k clusters** based on the similarity between points. The algorithm iteratively updates the positions of cluster centroids to minimize the distance between points and their centroids.
- After clustering, each point belongs to the cluster with the nearest centroid.
- **Hyperparameters** (such as the number of clusters, number of initializations, and the maximum number of iterations) can affect clustering quality, so they are optimized using **Optuna**.

### How It Works

#### Data Preparation
- Copies of the original and processed datasets are created to avoid changes to the original data.
- Columns are added to the dataset to store distances to the centroids and the cluster numbers.

#### Hyperparameter Optimization for K-Means
- Hyperparameters of K-Means such as the number of clusters (n_clusters), number of initializations (n_init), and maximum number of iterations (max_iter) are optimized using **Optuna**.
- Optimization is based on evaluating clustering quality using the **Calinski-Harabasz** score, which measures cluster density and separation.

#### Anomaly Selection
- After optimization, clustering is performed, and the distance to the centroid of its cluster is calculated for each record.
- To form the anomaly sample, the most "distant" records are selected uniformly from each cluster. This takes into account varying cluster densities and avoids favoring low-density clusters.
  - The **number of anomalies per cluster** can be explicitly set via the `anomalies_per_cluster` parameter. If not specified, the selection is distributed evenly between clusters in proportion to their size.
  - If the total number of selected anomalies exceeds the requested sample size, the number of anomalies per cluster is adjusted accordingly.

#### Sample Formation
- Selected records are added to the final sample, and these records are marked in the `is_sample` column.
- If the requested sample size exceeds the number of records, a warning is displayed, and the entire population is used.

#### Execution Report
- A summary report of the performed sampling is generated, including the number of clusters, model parameters, date and time of execution, sample size, and optimized hyperparameters.

---

## Important Points

- **Optuna-based Optimization**: K-Means hyperparameters are optimized using Optuna to improve clustering quality.
- **Uniform Anomaly Selection from Clusters**: Records are selected based on their distance to centroids, ensuring a uniform selection of anomalies from each cluster, regardless of density.
- **Reproducibility**: Using a random seed (`random_seed`) ensures reproducibility of clustering and sampling results.
- **Flexible Sampling**: The number of anomalies per cluster (`anomalies_per_cluster`) can be specified, or the sample can be distributed automatically between clusters.

---

## Potential Issues

- **Low Number of Clusters**: If the number of clusters is too small, it may lead to poor clustering, which will affect the quality of anomaly selection.
- **Insufficient Anomalies in a Cluster**: In clusters with very few records, it may be difficult to select the required number of anomalies. In this case, you can set a minimum or maximum number of anomalies per cluster.
- **Large Distances May Not Be Ideal**: Sampling based on the largest distances may not always meet the sample's goal in every specific case, especially if noisy points are present in the data.

---

## Conclusion

This script uses **K-Means clustering** with hyperparameter optimization via **Optuna**. Anomaly sampling is based on the distances to cluster centroids, accounting for cluster density to help highlight the most representative or anomalous records evenly from all clusters.

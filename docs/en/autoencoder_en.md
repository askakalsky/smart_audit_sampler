# Script Description

This script performs **anomaly-based sampling** using a **flexible autoencoder architecture**. The autoencoder is used to detect outliers in the data based on reconstruction error — the higher the error, the more anomalous the record is considered.

## How the autoencoder works:

### Basics of the autoencoder:
- An **autoencoder** is a type of neural network that learns to compress (encode) input data into a smaller space (bottleneck layer) and then reconstruct it (decode) back. The goal of the autoencoder is to reconstruct the original data as accurately as possible.
- The **flexible autoencoder** consists of two parts: the **encoder** and the **decoder**. The encoder compresses the input data into a hidden layer and then further into the bottleneck, while the decoder reconstructs the data back to its original size.
- For anomaly detection, the autoencoder is used to identify records that have large reconstruction errors, as these records do not conform to "normal" patterns in the data.

### How it works:

### Data Preparation:
- The original and preprocessed datasets are converted into **PyTorch tensors** for processing by the autoencoder.
- The data is passed through a **DataLoader** for batch processing during training.

### Flexible Autoencoder Architecture:
- The autoencoder model consists of two stages: the **encoder** (compresses data to the bottleneck) and the **decoder** (reconstructs data back to the original size).
- **LeakyReLU** activation layers are used to enhance training performance, and **Xavier initialization** is applied to the weights.

### Model Training:
- The model is trained using the preprocessed data for several epochs with the **Adam optimizer** and **MSELoss** to minimize reconstruction error.
- After each epoch, the average loss is computed and logged to assess progress.

### Anomaly Detection:
- After training, the autoencoder uses the reconstruction error for each record as the **anomaly score**. Records with the highest reconstruction errors are considered the most anomalous.
- All records are sorted by anomaly score, and the top records with the highest scores are selected for the sample.

### Forming the Sample:
- Both the original and preprocessed datasets receive additional columns for storing anomaly scores and sample labels.
- The most anomalous records are selected and added to the final sample.

### Execution Report:
- A summary report of the sampling process is generated, including the autoencoder architecture, sample size, number of records processed, and the random seed for reproducibility.

---

## Important points:

- **Flexible architecture**: The autoencoder has a bottleneck layer to compress data and a decoder to reconstruct it. This reduces dimensionality and identifies important patterns.
- **Anomaly detection**: Records with high reconstruction errors are considered anomalous, as the model cannot accurately reconstruct them.
- **Reproducibility**: Using a random seed ensures that the model and sampling results can be reproduced.

---

## Potential deviations:

- **Training parameters**: Adjusting the number of epochs or hidden layer sizes can affect the quality of anomaly detection.
- **Small samples**: If the sample size is too small, the model may not effectively distinguish normal data from anomalous data.

---

## In conclusion:

- **Flexible autoencoder** is a powerful tool for detecting anomalies in complex datasets. It uses reconstruction error to identify anomalous records, allowing for sample selection based on the most "unusual" data.

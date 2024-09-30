import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FlexibleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, bottleneck_dim=3):
        """
        Flexible autoencoder architecture with an additional hidden layer.

        Parameters:
        -----------
        input_dim : int
            Number of input features.
        hidden_dim : int, optional
            Size of the hidden layer. Default is 64.
        bottleneck_dim : int, optional
            Size of the bottleneck layer. Default is 3.
        """
        super(FlexibleAutoencoder, self).__init__()

        # Encoder: compress input to hidden layer, then to bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Compress input to hidden size
            nn.LeakyReLU(negative_slope=0.2),  # Activation
            nn.Linear(hidden_dim, bottleneck_dim),  # Compress to bottleneck
            nn.LeakyReLU(negative_slope=0.2)   # Activation
        )

        # Decoder: reconstruct from bottleneck to hidden, then back to input size
        self.decoder = nn.Sequential(
            # Expand from bottleneck to hidden size
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),  # Activation
            nn.Linear(hidden_dim, input_dim),   # Expand back to input size
            # Output activation to keep the range [0, 1] if necessary
            nn.Sigmoid()
        )

        # Xavier initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Forward pass: encode and then decode
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def autoencoder_sampling(data: pd.DataFrame, data_preprocessed: pd.DataFrame, sample_size: int,
                         features: list, random_seed: int):
    """
    Anomaly detection using flexible autoencoder sampling.

    Parameters:
    -----------
    data : pd.DataFrame
        Original dataset to be processed.
    data_preprocessed : pd.DataFrame
        Preprocessed dataset used for training the autoencoder.
    sample_size : int
        Number of samples to return in the final selection.
    features : list
        List of feature names, including both numerical and categorical features.
    random_seed : int
        Random seed for reproducibility.

    Returns:
    --------
    Tuple containing:
        - population_with_results : pd.DataFrame
          Original dataset with added fields "is_sample" and "anomaly_score".
        - population_for_chart : pd.DataFrame
          Preprocessed dataset with added fields "is_sample" and "anomaly_score".
        - sample : pd.DataFrame
          Sampled data where "is_sample" equals 1, with size equal to `sample_size`.
        - method_description : str
          Description of the autoencoder architecture and its details.
    """
    try:
        # Make copies to avoid modifying the original DataFrames
        data = data.copy()
        data_preprocessed = data_preprocessed.copy()

        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Convert preprocessed data to PyTorch tensor
        data_tensor = torch.tensor(
            data_preprocessed.values, dtype=torch.float32)

        # Create DataLoader for batch processing
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Initialize autoencoder model
        input_dim = data_preprocessed.shape[1]
        autoencoder = FlexibleAutoencoder(input_dim=input_dim)

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Training the autoencoder
        num_epochs = 10  # More epochs for deeper model
        logger.info(f"Training started with {num_epochs} epochs.")

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                batch_data = batch[0]  # Get batch data

                # Forward pass
                reconstructed = autoencoder(batch_data)
                loss = criterion(reconstructed, batch_data)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.6f}")

        # Detecting anomalies (Reconstruction error as anomaly score)
        autoencoder.eval()  # Set model to evaluation mode
        with torch.no_grad():
            reconstructed_data = autoencoder(data_tensor)
            reconstruction_errors = torch.mean(
                (reconstructed_data - data_tensor) ** 2, dim=1)
            anomaly_scores = reconstruction_errors.numpy()

        # Add anomaly scores to the original dataset
        population_with_results = data.copy()
        population_with_results['anomaly_score'] = anomaly_scores

        # Add anomaly scores to the preprocessed dataset (for charting)
        population_for_chart = data_preprocessed.copy()
        population_for_chart['anomaly_score'] = anomaly_scores

        # Sort data by anomaly score and select top "sample_size" records as the sample
        sample_indices = np.argsort(anomaly_scores)[-sample_size:]
        population_with_results['is_sample'] = 0
        population_with_results.loc[sample_indices, 'is_sample'] = 1

        population_for_chart['is_sample'] = 0
        population_for_chart.loc[sample_indices, 'is_sample'] = 1

        sample = population_with_results[population_with_results['is_sample'] == 1]

        # Total population size and number of records processed
        total_population_size = len(data)
        num_features = len(features)

        # Method description for logging
        method_description = (
            f"Autoencoder architecture: Flexible with 2 hidden layers (LeakyReLU activation), "
            f"input_dim = {input_dim}, hidden_dim = 64, bottleneck_dim = 3.\n"
            f"Trained on {total_population_size} records with {num_features} features.\n"
            f"Sample size = {sample_size}, random_seed = {random_seed}.\n"
            f"Detected top {sample_size} most anomalous records based on reconstruction error."
        )

        return population_with_results, population_for_chart, sample, method_description

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"ValueError: {ve}"

    except KeyError as ke:
        logger.error(f"KeyError: {ke}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"KeyError: {ke}"

    except Exception as e:
        logger.exception(f"Unexpected error in autoencoder_sampling: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Unexpected error: {e}"

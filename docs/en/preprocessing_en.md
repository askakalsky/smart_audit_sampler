# Script Description

This script defines a function `preprocess_data` used for data preprocessing. It performs several data cleaning and transformation tasks, such as one-hot encoding of categorical features, scaling of numerical data, and handling skewed features with logarithmic transformation. The script also configures logging to track the steps executed during preprocessing.

## Script Structure

### Imported Libraries:
- `pandas`, `numpy`: For data manipulation and numerical operations.
- `sklearn.preprocessing`: For scaling data using `StandardScaler` and `MinMaxScaler`.
- `scipy.stats`: For calculating skewness in numerical features.
- `logging`: For logging the progress and events during the execution.
- `datetime`: For recording timestamps of preprocessing steps.

### Main Function: `preprocess_data`

The `preprocess_data` function takes the following input parameters:
- `df`: A Pandas DataFrame containing the raw data to be processed.
- `numerical_columns`: A list of column names representing numerical features that need scaling or transformation.
- `categorical_columns`: A list of column names representing categorical features that will be one-hot encoded.
- `sample_fraction` (optional): A fraction of the dataset to use (default is 1.0, meaning the entire dataset is used).
- `skew_threshold` (optional): A threshold for skewness, above which log transformation is applied (default is 0.75).
- `random_seed` (optional): A seed for reproducibility when sampling data.

### Preprocessing Steps:

1. **Data Sampling (if needed):**  
   The function samples the data based on the `sample_fraction` parameter. If the fraction is less than 1.0, a subset of the data is randomly sampled. A log entry is made to record the sampling process.

2. **One-Hot Encoding of Categorical Features:**  
   Categorical columns specified by `categorical_columns` are transformed into binary indicator columns using one-hot encoding. The transformation generates new columns, which are tracked and logged.

3. **Handling Skewed Numerical Data:**  
   For numerical columns, skewness is calculated using the `scipy.stats.skew` function. If the skewness of a column exceeds the `skew_threshold`, the script applies a logarithmic transformation to reduce the skewness.

4. **Scaling:**  
   - For log-transformed (skewed) features, `MinMaxScaler` is used to scale the values to a range between 0 and 1.
   - For normally distributed features (those with low skewness), `StandardScaler` is applied to standardize the data (zero mean and unit variance).

5. **Final Feature Selection:**  
   After processing, the script keeps only the transformed numerical and newly generated categorical features.

### Logging:
Throughout the process, log entries are generated to track key actions, such as:
- Sampling of the data.
- Application of one-hot encoding to categorical columns.
- Log transformation and scaling of numerical columns.

The function returns:
- `processed_df`: A DataFrame with processed features (scaled numerical and one-hot encoded categorical).
- `method_description`: A string describing each preprocessing step applied to the dataset.

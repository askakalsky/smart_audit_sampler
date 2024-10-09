# Smart Audit Sampler

This application provides a user-friendly interface for creating audit samples using various statistical and machine learning-based methods. It supports multiple input file formats (CSV, Excel, DBF, JSON, Parquet) and generates a sample along with a detailed PDF report including visualizations.

## Features

* **Statistical Sampling Methods:**
    * Random Sampling
    * Systematic Sampling
    * Stratified Sampling
    * Monetary Unit Sampling (MUS)
* **Machine Learning-Based Sampling Methods:**
    * Isolation Forest
    * Local Outlier Factor (LOF)
    * K-Means Clustering
    * Autoencoder
    * HDBSCAN
* **Data Preprocessing:** Handles numerical and categorical features with appropriate scaling and encoding techniques.
* **Visualization:** Generates charts for strata distribution, cumulative value analysis, and UMAP projections for ML methods.
* **Reporting:** Creates a comprehensive PDF report summarizing the sampling methodology, parameters, and results, including visualizations.
* **Multilingual Support:** Supports both Ukrainian and English languages.
* **Dark Theme UI:** Provides a modern and visually appealing user interface.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/askakalsky/smart_audit_sampler.git
    ```
2.  **Install required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application:**

    ```bash
    start.bat
    ```
2.  **Browse and select your data file.**
3.  **Choose a sampling method.**
4.  **Specify required parameters (sample size, strata column, etc.).**
5.  **For ML methods, define column types (numerical/categorical) before creating the sample.**
6.  **Click "Create Sample".**
7.  The sampled data (CSV) and a PDF report will be saved in the same directory as the input file.

## Documentation

Detailed documentation for each sampling method is available in the `docs` folder (in both English and Ukrainian).  These documents describe the methodology, parameters, and potential deviations for each method.  Additionally, `test_results.txt` provides performance metrics on various dataset sizes.

## Project Structure

*   `main.py`: Main application file.
*   `requirements.txt`: List of required Python packages.
*   `start.bat`: Batch file to launch the application.
*   `ml_sampling/`: Contains modules for machine learning-based sampling methods.
*   `statistical_sampling/`: Contains modules for statistical sampling methods.
*   `utils/`: Contains utility functions for visualization and preprocessing.
*   `docs/`: Contains documentation files for each sampling method.
*   `scoring_methods/`: Contains scoring methods for clustering evaluation (e.g., Silhouette).

## Contributing

Contributions are welcome!  Please feel free to submit issues or pull requests.

## License

[MIT License](LICENSE)
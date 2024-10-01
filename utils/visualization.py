import pandas as pd
import matplotlib.pyplot as plt
import umap
import logging
import pandas as pd
import matplotlib.pyplot as plt
import umap
from optuna.importance import get_param_importances
from typing import Optional, List

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_strata_chart(
    population: pd.DataFrame,
    sample: pd.DataFrame,
    strata_column: str,
    output_path: str,
    threshold: Optional[float] = None,
    value_column: Optional[str] = None
):
    """
    Creates a bar chart comparing the distribution of different strata in the population and the sample.

    Parameters:
    -----------
    population : pd.DataFrame
        The full dataset containing the population.
    sample : pd.DataFrame
        The sampled subset of the population.
    strata_column : str
        The column in the DataFrame that defines the strata.
    output_path : str
        The path where the output chart image will be saved.
    threshold : Optional[float], optional
        If provided, filters the population by this threshold on the `value_column`.
    value_column : Optional[str], optional
        The column on which the threshold will be applied.

    Returns:
    --------
    None
        The bar chart is saved to the specified output path.
    """
    # Apply threshold to population if provided
    if threshold is not None and value_column is not None:
        population_filtered = population[population[value_column] >= threshold]
    else:
        population_filtered = population.copy()

    # Compute strata statistics on the filtered population
    population_strata_stats = population_filtered[strata_column].value_counts(
        normalize=True) * 100
    sample_strata_stats = sample[strata_column].value_counts(
        normalize=True) * 100

    # Ensure that all strata are represented
    sorted_strata = population_strata_stats.sort_values(ascending=False).index

    population_percentages = [population_strata_stats[stratum]
                              for stratum in sorted_strata]
    sample_percentages = [sample_strata_stats.get(
        stratum, 0) for stratum in sorted_strata]

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    x_coords = range(len(sorted_strata))

    # Plotting the population and sample strata side by side
    plt.bar([x - bar_width/2 for x in x_coords], population_percentages,
            bar_width, label='Population', color='blue', alpha=0.7)
    plt.bar([x + bar_width/2 for x in x_coords], sample_percentages,
            bar_width, label='Sample', color='red', alpha=0.7)

    plt.xlabel('Strata')
    plt.ylabel('Percentage')
    plt.title('Comparison of Strata Sizes in Population and Sample')
    plt.grid(True)
    plt.xticks(x_coords, sorted_strata, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()


def create_cumulative_chart(
    population: pd.DataFrame,
    value_column: str,
    strata_column: Optional[str],
    output_path: str,
    threshold: Optional[float] = None
):
    """
    Creates a cumulative chart based on monetary values in the population and highlights selected samples.

    Parameters:
    -----------
    population : pd.DataFrame
        The full dataset containing the population.
    value_column : str
        The column in the DataFrame representing the monetary value to accumulate.
    strata_column : Optional[str], optional
        The column representing strata. If provided, stratified cumulative charts will be created.
    output_path : str
        The path where the output chart image will be saved.
    threshold : Optional[float], optional
        If provided, filters the population by this threshold on the `value_column`.

    Returns:
    --------
    None
        The cumulative chart is saved to the specified output path.
    """
    # Apply threshold to population if provided
    if threshold is not None:
        population_filtered = population[population[value_column] >= threshold].copy(
        )
    else:
        population_filtered = population.copy()

    # Check if the filtered population is empty
    if population_filtered.empty:
        logger.warning(
            "The filtered population is empty after applying the threshold. No chart will be generated.")
        return

    # Create stratified charts or a single overall chart
    if strata_column and strata_column in population_filtered.columns:
        for stratum, group in population_filtered.groupby(strata_column):
            # Check if the group is empty
            if group.empty:
                logger.warning(
                    f"Stratum '{stratum}' is empty after applying the threshold. Skipping this stratum.")
                continue

            plt.figure(figsize=(10, 6))

            sorted_values = group.sort_values(
                by=value_column).reset_index(drop=True)
            cumulative_sums = sorted_values[value_column].cumsum()

            # Check if there are values to plot
            if sorted_values.empty:
                logger.warning(
                    f"No values to plot for stratum '{stratum}'. Skipping this stratum.")
                plt.close()
                continue

            # Plot cumulative sums
            plt.plot(range(len(sorted_values)),
                     cumulative_sums, label='Cumulative Sum')

            # Highlight sample points (where is_sample == 1)
            sample_indices = sorted_values[sorted_values['is_sample'] == 1].index

            # Plot sample points if any
            if len(sample_indices) > 0:
                plt.scatter(
                    sample_indices, cumulative_sums.iloc[sample_indices], color='red', label='Selected Items')
            else:
                logger.warning(
                    f"No matching sample values found in stratum '{stratum}' for plotting.")

            plt.xlabel('Elements in Stratum (sorted by monetary value)')
            plt.ylabel('Cumulative Sum of Monetary Values')
            plt.title(f'Cumulative Chart for Stratum {stratum}')
            plt.grid(True)
            plt.legend()

            stratum_output_path = f"{output_path}_stratum_{stratum}.png"
            plt.savefig(stratum_output_path)
            plt.close()

            logger.info(
                f"Saved cumulative chart for stratum {stratum}: {stratum_output_path}")
    else:
        plt.figure(figsize=(10, 6))

        sorted_values = population_filtered.sort_values(
            by=value_column).reset_index(drop=True)
        cumulative_sums = sorted_values[value_column].cumsum()

        # Check if there are values to plot
        if sorted_values.empty:
            logger.warning(
                "No values to plot for the overall population. No chart will be generated.")
            plt.close()
            return

        # Plot cumulative sums
        plt.plot(range(len(sorted_values)),
                 cumulative_sums, label='Cumulative Sum')

        # Highlight sample points (where is_sample == 1)
        sample_indices = sorted_values[sorted_values['is_sample'] == 1].index

        # Plot sample points if any
        if len(sample_indices) > 0:
            plt.scatter(
                sample_indices, cumulative_sums.iloc[sample_indices], color='red', label='Selected Items')
        else:
            logger.warning("No matching sample values found for plotting.")

        plt.xlabel('Elements (sorted by monetary value)')
        plt.ylabel('Cumulative Sum of Monetary Values')
        plt.title('Cumulative Chart of Monetary Unit Sampling')
        plt.legend()

        plt.savefig(output_path)
        plt.close()

        logger.info(f"Saved overall cumulative chart: {output_path}")


def create_umap_projection(
        population: pd.DataFrame,
        label_column: str,
        features: List[str],
        output_path: str,
        cluster_column: Optional[str] = None):
    """
    Creates a UMAP projection to visualize high-dimensional data and labels (anomalies or clusters).

    Parameters:
    -----------
    population : pd.DataFrame
        The dataset containing the features and labels to be visualized.
    label_column : str
        The column representing the labels (e.g., anomalies) for coloring the data points.
    features : List[str]
        The list of features to include in the UMAP projection.
    output_path : str
        The path where the UMAP plot will be saved.
    cluster_column : Optional[str], optional
        Optional column representing cluster assignments. If provided, clusters will be shown on the plot.

    Returns:
    --------
    None
        The UMAP projection chart is saved to the specified output path.
    """
    population = population.copy()

    # Check if the label column is present
    if label_column not in population.columns:
        raise ValueError(f"Column '{label_column}' not found in the DataFrame")

    # Check for cluster column and ensure it is not in features
    has_cluster_column = cluster_column and cluster_column in population.columns

    if has_cluster_column:
        if cluster_column in features:
            raise ValueError(
                f"Column '{cluster_column}' should not be in the features list")
        population = population.dropna(subset=[label_column, cluster_column])
    else:
        population = population.dropna(subset=[label_column])

    available_features = get_available_features(population, features)
    feature_data = population[available_features]

    if feature_data.empty:
        raise ValueError("No features available for UMAP projection")

    # Perform UMAP reduction
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(feature_data)

    labels = population[label_column].values

    # Masks for anomalies (label = 1) and clusters (label = 0)
    mask_anomalies = labels == 1
    mask_clusters = labels == 0

    plt.figure(figsize=(10, 8))

    # Plot clusters if cluster column is available
    if has_cluster_column and mask_clusters.sum() > 0:
        clusters = population[cluster_column].values
        scatter = plt.scatter(
            embedding[mask_clusters, 0], embedding[mask_clusters, 1],
            c=clusters[mask_clusters], cmap='viridis', s=10, alpha=0.7, label='Clustered (label=0)')
        plt.colorbar(scatter, label=cluster_column)
    else:
        plt.scatter(
            embedding[mask_clusters, 0], embedding[mask_clusters, 1],
            facecolor='blue', label='Clustered (label=0)',
            s=10, alpha=0.7)

    # Plot anomalies (label = 1)
    if mask_anomalies.sum() > 0:
        plt.scatter(
            embedding[mask_anomalies, 0], embedding[mask_anomalies, 1],
            facecolor='white', edgecolor='black', label='Anomalies (label=1)',
            s=50, linewidth=1.5, alpha=1.0)

    plt.title(f'UMAP Projection (anomalies in black)' if not has_cluster_column else
              f'UMAP Projection (colored by {cluster_column}, anomalies in white with black edge)')

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    plt.savefig(output_path)
    plt.close()

    logger.info(f"UMAP plot saved to: {output_path}")


def plot_optimization_history(study, output_dir):
    """
    Plots the optimization history of an Optuna study.

    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object containing the optimization results.
    output_dir : str
        Directory path where the optimization history plot will be saved.

    Returns:
    --------
    None
        The optimization history plot is saved to the specified output directory.
    """
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]

    plt.figure()
    plt.plot(values, marker='o')
    plt.xlabel('Trial')
    plt.ylabel('Objective Value')
    plt.title('Optimization History')
    plt.savefig(f"{output_dir}_optimization_history.png")
    plt.close()


def plot_param_importances(study, output_dir):
    """
    Plots the parameter importances from an Optuna study.

    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object containing the parameter importances.
    output_dir : str
        Directory path where the parameter importance plot will be saved.

    Returns:
    --------
    None
        The parameter importance plot is saved to the specified output directory.
    """
    importances = get_param_importances(study)
    params = list(importances.keys())
    values = list(importances.values())

    plt.figure()
    plt.barh(params, values)
    plt.xlabel('Importance')
    plt.title('Parameter Importances')
    plt.savefig(f"{output_dir}_param_importances.png")
    plt.close()


def plot_parallel_coordinate(study, output_dir):
    """
    Plots the parallel coordinate plot for the Optuna study results.

    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object containing the trial data.
    output_dir : str
        Directory path where the parallel coordinate plot will be saved.

    Returns:
    --------
    None
        The parallel coordinate plot is saved to the specified output directory.
    """
    trials = study.trials
    df = pd.DataFrame([{
        'trial': t.number,
        'value': t.value,
        **t.params
    } for t in trials if t.value is not None])

    pd.plotting.parallel_coordinates(df, 'value', colormap='viridis')
    plt.title('Parallel Coordinate Plot')
    plt.savefig(f"{output_dir}_parallel_coordinate.png")
    plt.close()


def plot_slice(study, output_dir):
    """
    Plots the slice plot for each parameter in the Optuna study results.

    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object containing the trial data.
    output_dir : str
        Directory path where the slice plots will be saved.

    Returns:
    --------
    None
        The slice plots are saved to the specified output directory.
    """
    trials = study.trials
    df = pd.DataFrame([{
        'trial': t.number,
        'value': t.value,
        **t.params
    } for t in trials if t.value is not None])

    for param in df.columns[2:]:
        plt.figure()
        plt.scatter(df[param], df['value'], alpha=0.5)
        plt.xlabel(param)
        plt.ylabel('Objective Value')
        plt.title(f'Slice Plot for {param}')
        plt.savefig(f"{output_dir}_slice_{param}.png")
        plt.close()


def visualize_optuna_results(study, output_dir):
    """
    Generates and saves a set of visualization plots for an Optuna study.

    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object containing the optimization results.
    output_dir : str
        Directory path where the plots will be saved.

    Returns:
    --------
    None
        All generated plots are saved to the specified output directory.
    """
    try:
        plot_optimization_history(study, output_dir)
        plot_param_importances(study, output_dir)
        plot_parallel_coordinate(study, output_dir)
        plot_slice(study, output_dir)

        logger.info(f"Visualization results saved in {output_dir}")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")


def get_available_features(population: pd.DataFrame, features: List[str]) -> List[str]:
    """
    Identifies and returns the available features from the given list in the DataFrame.

    Parameters:
    -----------
    population : pd.DataFrame
        The dataset containing the columns.
    features : List[str]
        List of features to check for availability in the DataFrame.

    Returns:
    --------
    List[str]
        A list of available features found in the DataFrame.
    """

    available_features = [f for f in features if f in population.columns]
    missing_features = set(features) - set(available_features)
    for feature in missing_features:
        related_columns = [
            col for col in population.columns if col.startswith(f"{feature}_")]
        available_features.extend(related_columns)
    return list(dict.fromkeys(available_features))

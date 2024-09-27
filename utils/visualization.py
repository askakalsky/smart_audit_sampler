import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
import umap

# Standard libraries
import os
import random
import logging
import datetime
import uuid

# Data manipulation and analysis
import numpy as np
import pandas as pd
import dask.dataframe as dd

# Machine learning
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min, average_precision_score, mean_squared_error, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from skopt import BayesSearchCV

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

# Visualization
import matplotlib.pyplot as plt
import plotly.io as pio
import umap

# GUI
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# Optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import optuna.visualization as vis
from optuna.importance import get_param_importances

# Parallel processing
from joblib import Parallel, delayed
import joblib

# Type hinting
from typing import Optional, Tuple, List, Dict

# Process control
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_strata_chart(population: pd.DataFrame, sample: pd.DataFrame, strata_column: str, output_path: str):
    population_strata_stats = population[strata_column].value_counts(
        normalize=True) * 100
    sample_strata_stats = sample[strata_column].value_counts(
        normalize=True) * 100

    sorted_strata = population_strata_stats.sort_values(ascending=False).index

    population_percentages = [population_strata_stats[stratum]
                              for stratum in sorted_strata]
    sample_percentages = [sample_strata_stats.get(
        stratum, 0) for stratum in sorted_strata]

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    x_coords = range(len(sorted_strata))

    plt.bar([x - bar_width/2 for x in x_coords], population_percentages,
            bar_width, label='Популяція', color='blue', alpha=0.7)
    plt.bar([x + bar_width/2 for x in x_coords], sample_percentages,
            bar_width, label='Вибірка', color='red', alpha=0.7)

    plt.xlabel('Страти')
    plt.ylabel('Відсоток')
    plt.title('Порівняння розмірів страт у популяції та вибірці')
    plt.xticks(x_coords, sorted_strata, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()


def create_cumulative_chart(population: pd.DataFrame, sample: pd.DataFrame, value_column: str, strata_column: Optional[str], output_path: str):
    if strata_column and strata_column in population.columns:
        for stratum, group in population.groupby(strata_column):
            plt.figure(figsize=(10, 6))

            sorted_values = sorted(group[value_column])
            cumulative_sums = np.cumsum(sorted_values)

            plt.plot(range(len(sorted_values)), cumulative_sums,
                     label='Кумулятивна сума')

            sample_in_stratum = sample[sample[strata_column] == stratum]
            sample_indices = [sorted_values.index(
                val) for val in sample_in_stratum[value_column] if val in sorted_values]
            plt.scatter(sample_indices, [
                        cumulative_sums[i] for i in sample_indices], color='red', label='Вибрані елементи')

            plt.xlabel(
                'Елементи страти (відсортовані за грошовою величиною)')
            plt.ylabel('Кумулятивна сума грошових значень')
            plt.title(f'Кумулятивний графік для страти {stratum}')
            plt.legend()

            stratum_output_path = f"{output_path}_stratum_{stratum}.png"
            plt.savefig(stratum_output_path)
            plt.close()

            logger.info(
                f"Збережено кумулятивний графік для страти {stratum}: {stratum_output_path}")
    else:
        plt.figure(figsize=(10, 6))

        sorted_values = sorted(population[value_column])
        cumulative_sums = np.cumsum(sorted_values)

        plt.plot(range(len(sorted_values)), cumulative_sums,
                 label='Кумулятивна сума')

        sample_indices = [sorted_values.index(
            val) for val in sample[value_column] if val in sorted_values]
        plt.scatter(sample_indices, [
                    cumulative_sums[i] for i in sample_indices], color='red', label='Вибрані елементи')

        plt.xlabel('Елементи (відсортовані за грошовою величиною)')
        plt.ylabel('Кумулятивна сума грошових значень')
        plt.title('Кумулятивний графік методу грошової одиниці')
        plt.legend()

        plt.savefig(output_path)
        plt.close()

        logger.info(f"Збережено загальний кумулятивний графік: {output_path}")


def create_umap_projection(population: pd.DataFrame, label_column: str, features: List[str], output_path: str, random_seed: int):
    population = population.copy()
    available_features = get_available_features(population, features)
    feature_data = population[available_features]

    reducer = umap.UMAP(n_components=2, random_state=random_seed)
    embedding = reducer.fit_transform(feature_data)

    labels = population[label_column].values

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap='coolwarm', s=10, label=labels)

    plt.colorbar(scatter, ticks=[0, 1], label=label_column)

    plt.title(f'UMAP Projection (colored by {label_column})')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    plt.savefig(output_path)
    plt.close()

    logger.info(f"UMAP график сохранен по пути: {output_path}")


def plot_optimization_history(study, output_dir):
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
    try:
        plot_optimization_history(study, output_dir)
        plot_param_importances(study, output_dir)
        plot_parallel_coordinate(study, output_dir)
        plot_slice(study, output_dir)

        logger.info(f"Visualization results saved in {output_dir}")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")


def get_available_features(population: pd.DataFrame, features: List[str]) -> List[str]:
    available_features = [f for f in features if f in population.columns]
    missing_features = set(features) - set(available_features)
    for feature in missing_features:
        related_columns = [
            col for col in population.columns if col.startswith(f"{feature}_")]
        available_features.extend(related_columns)
    return list(dict.fromkeys(available_features))

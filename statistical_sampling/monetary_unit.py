import pandas as pd
import random
import uuid
from typing import Optional, Tuple, List, Dict

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


def monetary_unit_sampling(population: pd.DataFrame, sample_size: int, value_column: str,
                           threshold: Optional[float] = None, strata_column: Optional[str] = None,
                           random_seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    try:
        population = population.copy()
        population['unique_id'] = [uuid.uuid4()
                                   for _ in range(len(population))]
        population['is_sample'] = 0

        if threshold is not None:
            population_filtered = population[population[value_column] >= threshold].copy(
            )
        else:
            population_filtered = population.copy()

        if len(population_filtered) == 0:
            raise ValueError(
                "Після застосування порогу не залишилось записів.")

        population_filtered = population_filtered.reset_index(drop=True)

        def sample_stratum(group, stratum_sample_size, random_seed):
            if len(group) <= stratum_sample_size:
                return group
            group['CumulativeValue'] = group[value_column].cumsum()
            total_value = group['CumulativeValue'].iloc[-1]
            if total_value == 0:
                return group.sample(n=stratum_sample_size, random_state=random_seed)
            interval = total_value / stratum_sample_size
            random_gen = random.Random(random_seed)
            selection_points = [random_gen.uniform(
                0, interval) + i * interval for i in range(stratum_sample_size)]
            sample_indices = group['CumulativeValue'].searchsorted(
                selection_points)
            sample_indices = pd.Series(
                sample_indices).drop_duplicates().tolist()
            while len(sample_indices) < stratum_sample_size:
                new_point = random_gen.uniform(0, total_value)
                new_index = group['CumulativeValue'].searchsorted(new_point)
                if new_index not in sample_indices:
                    sample_indices.append(new_index)
            return group.iloc[sample_indices[:stratum_sample_size]]

        if strata_column:
            strata = population_filtered.groupby(strata_column)
            sample = pd.DataFrame()

            stratum_sizes = (strata.size() / len(population_filtered)
                             * sample_size).round().astype(int)
            size_diff = sample_size - stratum_sizes.sum()
            largest_strata = stratum_sizes.nlargest(abs(size_diff)).index

            for stratum in largest_strata:
                stratum_sizes[stratum] += 1 if size_diff > 0 else -1
                size_diff += -1 if size_diff > 0 else 1
                if size_diff == 0:
                    break

            for stratum, group in strata:
                stratum_sample_size = stratum_sizes[stratum]
                if stratum_sample_size == 0:
                    continue
                stratum_sample = sample_stratum(
                    group, stratum_sample_size, random_seed)
                sample = pd.concat([sample, stratum_sample])
        else:
            sample = sample_stratum(
                population_filtered, sample_size, random_seed)

        sample = sample.drop(columns=['CumulativeValue'], errors='ignore')
        sample['is_sample'] = 1

        sample_ids = sample['unique_id'].tolist()
        population.loc[population['unique_id'].isin(
            sample_ids), 'is_sample'] = 1

        population = population.drop('unique_id', axis=1)
        sample = sample.drop('unique_id', axis=1)

        method_description = f"Метод грошової одиниці. Розмір вибірки: {sample_size}. "
        method_description += f"Стовпець значень: {value_column}. "
        if threshold is not None:
            method_description += f"Порогове значення: {threshold}. "
        if strata_column:
            method_description += f"Стовпець стратифікації: {strata_column}. "
        method_description += f"Випадкове число: {random_seed}."

        return population, sample, method_description
    except Exception as e:
        return None, None, f"Помилка: {e}"

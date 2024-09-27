import pandas as pd
import random
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


def stratified_sampling(population: pd.DataFrame, sample_size: int, strata_column: str, random_seed: int) -> Tuple[pd.DataFrame, str]:
    try:
        population['is_sample'] = 0
        strata = population.groupby(strata_column)
        sample = pd.DataFrame()
        proportions = population[strata_column].value_counts(normalize=True)

        for stratum, proportion in proportions.items():
            stratum_sample_size = round(proportion * sample_size)
            if stratum_sample_size > 0:
                stratum_sample = strata.get_group(stratum).sample(n=min(
                    stratum_sample_size, len(strata.get_group(stratum))), random_state=random_seed)
                sample = pd.concat([sample, stratum_sample])

        while len(sample) < sample_size:
            largest_stratum = proportions.index[0]
            additional_sample = strata.get_group(
                largest_stratum).sample(n=1, random_state=random_seed)
            sample = pd.concat([sample, additional_sample])
            proportions[largest_stratum] -= 1 / len(population)
            proportions = proportions.sort_values(ascending=False)

        population.loc[sample.index, 'is_sample'] = 1
        sample['is_sample'] = population.loc[sample.index, 'is_sample'].values
        method_description = f"Стратифікована вибірка. Розмір вибірки: {sample_size}. Стовпець стратифікації: {strata_column}. Випадкове число: {random_seed}."
        return population, sample, method_description
    except Exception as e:
        logger.exception(f"Помилка у stratified_sampling: {e}")
        return None, f"Помилка: {e}"

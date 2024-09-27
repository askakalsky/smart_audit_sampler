import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
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


def lof_sampling(population_original: pd.DataFrame, population: pd.DataFrame, sample_size: int, features: List[str], random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, optuna.Study]:
    try:
        population_original = population_original.copy()
        population = population.copy()
        available_features = get_available_features(population, features)
        if not available_features:
            raise ValueError("Не знайдено відповідних ознак для аналізу")

        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 5, 50),
                'leaf_size': trial.suggest_int('leaf_size', 10, 100),
                'p': trial.suggest_int('p', 1, 3),
            }
            lof = LocalOutlierFactor(novelty=True, **params)
            lof.fit(population[available_features])

            return calculate_silhouette(lof, population, available_features, random_seed)

        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        best_params = study.best_params
        best_lof = LocalOutlierFactor(novelty=True, **best_params)
        best_lof.fit(population[available_features])

        population_original['anomaly_score'] = population['anomaly_score'] = best_lof.decision_function(
            population[available_features])
        population_original['is_anomaly'] = population['is_anomaly'] = (best_lof.predict(
            population[available_features]) == -1).astype(int)
        population_original['is_sample'] = population['is_sample'] = False

        population_original = population_original.sort_values(
            by='anomaly_score', ascending=True)
        population = population.sort_values(
            by='anomaly_score', ascending=True)

        sample_processed = population_original.head(sample_size)

        population_original.loc[sample_processed.index, 'is_sample'] = True
        population.loc[sample_processed.index, 'is_sample'] = True

        method_description = (
            f"Метод: Local Outlier Factor (LOF) з автоматичним підбором гіперпараметрів за допомогою Optuna.\n"
            f"Розмір вибірки: {sample_size}.\n"
            f"Запитані ознаки: {features}.\n"
            f"Використані ознаки: {available_features}.\n"
            f"Оптимальні параметри LOF: {best_params}.\n"
            f"Дата та час створення вибірки: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Кількість пробних параметрів: {len(study.trials)}.\n"
            f"Випадкове зерно: {random_seed}.\n"
        )

        return population_original, population, sample_processed, method_description, study

    except Exception as e:
        logger.exception(f"Помилка у lof_sampling: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Помилка: {e}", None

# Общая функция для расчета силуэтного коэффициента


def calculate_silhouette(clf, data: pd.DataFrame, available_features: List[str], random_seed: int) -> float:
    rng = np.random.default_rng(random_seed)
    n_samples = len(data)

    # Условие для определения размеров подвыборок
    if n_samples < 1000:
        sample_sizes = [n_samples]
    elif 1000 <= n_samples < 10000:
        sample_size = min(max(int(n_samples * 0.1), 1000), 5000)
        sample_sizes = [sample_size] * 3
    elif 10000 <= n_samples < 100000:
        sample_size = min(max(int(n_samples * 0.1), 1000), 5000)
        sample_sizes = [sample_size] * 3
    else:
        sample_size = 5000
        sample_sizes = [sample_size] * 3

    silhouette_scores = []
    for sample_size in sample_sizes:
        try:
            sample_indices = rng.choice(
                n_samples, size=sample_size, replace=False)
            sample = data.iloc[sample_indices]
            labels = clf.predict(data[available_features])
            score = silhouette_score(
                sample[available_features], labels[sample_indices])
            silhouette_scores.append(score)
        except ValueError:
            silhouette_scores.append(float('-inf'))

    return np.mean(silhouette_scores)


def get_available_features(population: pd.DataFrame, features: List[str]) -> List[str]:
    available_features = [f for f in features if f in population.columns]
    missing_features = set(features) - set(available_features)
    for feature in missing_features:
        related_columns = [
            col for col in population.columns if col.startswith(f"{feature}_")]
        available_features.extend(related_columns)
    return list(dict.fromkeys(available_features))

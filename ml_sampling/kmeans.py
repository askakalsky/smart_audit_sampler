import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import optuna
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


def kmeans_sampling(population_original: pd.DataFrame, population: pd.DataFrame, sample_size: int, features: List[str], random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    try:
        population_original = population_original.copy()
        population = population.copy()
        available_features = get_available_features(population, features)
        if not available_features:
            raise ValueError("Не знайдено відповідних ознак для аналізу")

        X = population[available_features]
        if X.isnull().values.any():
            raise ValueError("Набір даних містить пропущені значення.")

        def objective(trial):
            params = {
                'n_clusters': trial.suggest_int('n_clusters', 2, 20),
                'init': trial.suggest_categorical('init', ['k-means++', 'random']),
                'n_init': trial.suggest_int('n_init', 5, 30),
                'max_iter': trial.suggest_int('max_iter', 100, 500),
            }
            kmeans = KMeans(random_state=random_seed, **params)
            kmeans.fit(X)
            score = silhouette_score(X, kmeans.labels_)
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, show_progress_bar=True)

        best_params = study.best_params
        best_kmeans = KMeans(random_state=random_seed, **best_params)
        best_kmeans.fit(X)
        clusters = best_kmeans.predict(X)
        if not np.issubdtype(clusters.dtype, np.integer):
            clusters = clusters.astype(int)

        population_original['cluster'] = population['cluster'] = clusters
        logger.debug(f"Кластери (type: {clusters.dtype}): {clusters[:10]}")
        logger.debug(
            f"Центроиды кластеров (shape: {best_kmeans.cluster_centers_.shape}): {best_kmeans.cluster_centers_[:2]}")
        logger.debug(f"Расстояния до центроида: {len(X)} значений")
        _, distances = pairwise_distances_argmin_min(
            X, best_kmeans.cluster_centers_)
        population_original['distance_to_centroid'] = population['distance_to_centroid'] = distances
        logger.debug(f"Расстояния до центроида (пример): {distances[:10]}")
        population_original = population_original.sort_values(
            'distance_to_centroid', ascending=False)
        population = population.sort_values(
            'distance_to_centroid', ascending=False)
        if len(population_original) < sample_size:
            logger.warning(
                f"Розмір вибірки {len(population_original)} менший за запитуваний {sample_size}. Вибираються всі доступні записи.")
            sample_size = len(population_original)

        sample_processed = population_original.head(sample_size)
        population_original['is_sample'] = population['is_sample'] = False
        population_original.loc[sample_processed.index, 'is_sample'] = True
        population.loc[sample_processed.index, 'is_sample'] = True

        method_description = (
            f"Вибірка на основі кластеризації K-Means з автоматичним підбором гіперпараметрів (Optuna).\n"
            f"Розмір вибірки: {sample_size}.\n"
            f"Запитані ознаки: {features}.\n"
            f"Використані ознаки: {available_features}.\n"
            f"Оптимальні параметри K-Means: {best_params}.\n"
            f"Дата та час створення вибірки: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Випадкове зерно: {random_seed}.\n"
            f"Кількість пробних параметрів: {len(study.trials)}.\n"
        )

        return population_original, population, sample_processed, method_description

    except Exception as e:
        logger.exception(f"Помилка у kmeans_sampling: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Помилка: {e}"


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

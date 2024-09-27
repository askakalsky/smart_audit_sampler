from tqdm import tqdm
import joblib
import optuna.visualization as vis
from optuna.pruners import MedianPruner
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
import optuna
import copy  # Добавлено для копирования лучшей модели
from typing import Optional, Tuple, List, Dict
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def create_dataloader(data, batch_size, shuffle=True):
    return DataLoader(TensorDataset(torch.from_numpy(data.values).float()),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def create_and_train_model(trial, X_train, X_val, device, max_epochs=50):
    input_dim = X_train.shape[1]
    input_size = X_train.shape[0]

    min_batch_size = determine_batch_size(input_size)
    max_batch_size = min_batch_size * 2

    hidden_dim = trial.suggest_int(
        'hidden_dim', int(input_dim * 0.5), int(input_dim * 1.25))
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int(
        'batch_size', min_batch_size, max_batch_size)

    logger.info(f"Creating and training model with hidden_dim={hidden_dim}, "
                f"learning_rate={learning_rate}, batch_size={batch_size}")

    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    use_amp = torch.cuda.is_available()

    train_loader = create_dataloader(X_train, batch_size)
    val_loader = create_dataloader(X_val, batch_size)

    best_val_loss = float('inf')
    best_model = None  # Для сохранения наилучшей модели
    patience = 10
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        val_loss = evaluate_model(model, val_loader, criterion, device)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience = 10
        else:
            patience -= 1
            if patience == 0:
                break

    return best_model, best_val_loss


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    use_amp = torch.cuda.is_available()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
            total_loss += loss.item() * inputs.size(0)
    mean_loss = total_loss / len(dataloader.dataset)
    return mean_loss


def objective_with_cv(trial, X, n_splits=3, device='cuda', random_seed=None):
    logger.debug(f"Starting cross-validation with {n_splits} splits.")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    cv_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        model, val_score = create_and_train_model(
            trial, X_train, X_val, device)
        cv_scores.append(val_score)

    mean_score = np.mean(cv_scores)
    logger.info(f"Cross-validation mean score: {mean_score:.16f}")
    return mean_score


def optimize_autoencoder(X: np.ndarray, n_trials: int = 30, n_restarts: int = 3, random_seed: int = None) -> Tuple[dict, float, optuna.Study]:
    logger.info(
        f"Starting optimization with {n_trials} trials and {n_restarts} restarts.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_study = None
    best_value = float('inf')

    for restart in tqdm(range(n_restarts), desc="Optimization restarts"):
        logger.info(f"Optimization restart {restart+1}/{n_restarts}")
        pruner = MedianPruner(n_startup_trials=5,
                              n_warmup_steps=10, interval_steps=1)
        study = optuna.create_study(direction='minimize', pruner=pruner)
        study.optimize(lambda trial: objective_with_cv(trial, X, device=device, random_seed=random_seed),
                       n_trials=n_trials // n_restarts, n_jobs=-1, show_progress_bar=True)

        if study.best_value < best_value:
            best_value = study.best_value
            best_study = study
            logger.info(f"New best value found: {best_value:.4f}")

    return best_study.best_params, best_value, best_study


def compute_reconstruction_errors(model, data, device, batch_size=1024):
    logger.debug(
        f"Computing reconstruction errors with batch_size={batch_size}")
    model.eval()
    dataloader = create_dataloader(data, batch_size, shuffle=False)
    reconstruction_errors = []
    use_amp = torch.cuda.is_available()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                errors = torch.mean((outputs - inputs) ** 2, dim=1)
            reconstruction_errors.extend(errors.cpu().numpy())
    return np.array(reconstruction_errors)


def autoencoder_sampling(population_original: pd.DataFrame,
                         population: pd.DataFrame,
                         sample_size: int,
                         features: List[str],
                         random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, optuna.Study]:
    try:
        logger.info(
            f"Starting autoencoder sampling with sample_size={sample_size}, random_seed={random_seed}")
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        scaled_data = population

        subsample_size = min(100000, len(scaled_data))
        subsample_indices = np.random.choice(
            len(scaled_data), subsample_size, replace=False)
        subsample_data = scaled_data.iloc[subsample_indices]

        best_params, best_value, best_study = optimize_autoencoder(
            subsample_data, random_seed=random_seed)
        logger.info(
            f"Best parameters: {best_params}, Best value: {best_value:.16f}")

        final_model = Autoencoder(
            len(scaled_data.columns), best_params['hidden_dim']).to(device)
        optimizer = optim.Adam(final_model.parameters(),
                               lr=best_params['learning_rate'])
        criterion = nn.MSELoss()
        scaler = GradScaler()
        use_amp = torch.cuda.is_available()

        full_train_loader = create_dataloader(
            scaled_data, batch_size=best_params['batch_size'])

        for epoch in range(100):
            final_model.train()
            for batch in full_train_loader:
                inputs = batch[0].to(device)
                optimizer.zero_grad()
                with autocast(enabled=use_amp):
                    outputs = final_model(inputs)
                    loss = criterion(outputs, inputs)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        reconstruction_errors = compute_reconstruction_errors(
            final_model, scaled_data, device)

        population_original['reconstruction_error'] = reconstruction_errors
        population['reconstruction_error'] = reconstruction_errors

        population_original = population_original.sort_values(
            by='reconstruction_error', ascending=False)
        population = population.sort_values(
            by='reconstruction_error', ascending=False)
        sample_processed = population_original.head(sample_size)

        population_original['is_sample'] = population['is_sample'] = False
        population_original.loc[sample_processed.index, 'is_sample'] = True
        population.loc[sample_processed.index, 'is_sample'] = True

        torch.save(final_model.state_dict(), 'autoencoder_model.pth')
        joblib.dump(scaler, 'scaler.joblib')
        logger.info(f"Model and scaler saved successfully.")

        method_description = (
            f"Вибірка на основі Autoencoder з автоматичним підбором гіперпараметрів (Optuna).\n"
            f"Розмір вибірки: {sample_size}.\n"
            f"Найкраще значення цільової функції: {best_value}.\n"
            f"Requested features: {features}.\n"
            f"Кількість аномалій: {len(sample_processed)}.\n"
            f"Дата та час створення вибірки: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Випадкове зерно: {random_seed}.\n"
        )

        return population_original, population, sample_processed, method_description, best_study

    except ValueError as e:
        logger.error(f"ValueError: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"ValueError: {e}", None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Unexpected error: {e}", None


def determine_batch_size(input_size):
    if input_size < 1000:
        return 16
    elif 1000 <= input_size < 5000:
        return 32
    elif 5000 <= input_size < 10000:
        return 48
    elif 10000 <= input_size < 30000:
        return 64
    elif 30000 <= input_size < 100000:
        return 128
    else:
        return 256

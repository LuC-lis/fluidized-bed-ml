"""
Seeded, reproducible evaluation harness.

Scaler fitting, the k-fold split, early stopping, per-fold metrics and
averaging are factored out here, so a result is one function call and one
configuration object.

Guarantees
----------
* The scaler is fitted **inside** each fold, so no information leaks from the
  validation split into training.
* All random sources (numpy, torch, python) are seeded from one integer.
* ``train_one_fold`` never sees the validation data during training.
* Every result is reported per fold and as an aggregate, so the spread across
  splits is visible rather than hidden behind a single number.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Everything that influences a single training run."""

    seed: int = 42
    learning_rate: float = 0.005
    weight_decay: float = 0.4
    batch_size: int = 72
    num_epochs: int = 300
    patience: int = 50
    min_delta: float = 0.0
    clip_grad_norm: float = 1.0
    loss: str = "mse"          # "mse" | "smooth_l1"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    standardize_targets: bool = True
    n_splits: int = 5
    verbose: bool = False
    device: str = "cpu"

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# one fold
# ---------------------------------------------------------------------------
def train_one_fold(
    model_factory: Callable[[], nn.Module],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> Dict[str, np.ndarray]:
    """Train a model on one split and return predictions on train and validation.

    Returns
    -------
    dict with ``y_val``, ``y_val_pred``, ``y_train``, ``y_train_pred`` (all in
    the *original* target units) and ``history`` (per-epoch losses).
    """
    set_seed(config.seed)
    device = torch.device(config.device)

    feature_scaler = StandardScaler().fit(X_train)
    Xtr = feature_scaler.transform(X_train).astype(np.float32)
    Xva = feature_scaler.transform(X_val).astype(np.float32)

    if config.standardize_targets:
        target_scaler = StandardScaler().fit(y_train)
        ytr = target_scaler.transform(y_train).astype(np.float32)
    else:
        target_scaler = None
        ytr = np.asarray(y_train, dtype=np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=min(config.batch_size, len(Xtr)),
        shuffle=True,
    )

    model = model_factory().to(device)
    criterion = nn.MSELoss() if config.loss == "mse" else nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=config.scheduler_patience, factor=config.scheduler_factor
    )

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    trials = 0
    history: List[dict] = []

    def _predict(x_np: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(x_np).to(device)).cpu().numpy()
        return target_scaler.inverse_transform(out) if target_scaler is not None else out

    for epoch in range(config.num_epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            if config.clip_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_grad_norm)
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(Xtr)

        # validation loss in the (possibly standardised) space, used for early stopping
        model.eval()
        with torch.no_grad():
            pred_val = model(torch.from_numpy(Xva).to(device))
            if target_scaler is not None:
                yva_scaled = torch.from_numpy(
                    target_scaler.transform(y_val).astype(np.float32)
                ).to(device)
            else:
                yva_scaled = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).to(device)
            val_loss = float(criterion(pred_val, yva_scaled).item())

        scheduler.step(val_loss)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

        if val_loss < best_val - config.min_delta:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            trials = 0
        else:
            trials += 1
            if trials >= config.patience:
                if config.verbose:
                    print(f"    early stop at epoch {epoch} (best val loss {best_val:.6g})")
                break

    model.load_state_dict(best_state)
    return dict(
        y_train=np.asarray(y_train, dtype=float),
        y_train_pred=_predict(Xtr),
        y_val=np.asarray(y_val, dtype=float),
        y_val_pred=_predict(Xva),
        best_val_loss=best_val,
        epochs_run=len(history),
        history=history,
    )


# ---------------------------------------------------------------------------
# k-fold cross-validation
# ---------------------------------------------------------------------------
def cross_validate(
    model_factory: Callable[[], nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[TrainConfig] = None,
    target_names: Optional[Sequence[str]] = None,
    folds: Optional[Sequence] = None,
) -> Dict[str, object]:
    """Run k-fold cross-validation and return per-fold and aggregate metrics."""
    config = TrainConfig() if config is None else config
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    target_names = list(target_names) if target_names else [f"y{i}" for i in range(y.shape[1])]

    splitter = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    split_iterator = folds if folds is not None else splitter.split(X)

    fold_rows: List[dict] = []
    predictions: List[dict] = []
    for fold, (train_idx, val_idx) in enumerate(split_iterator, start=1):
        if config.verbose:
            print(f"  fold {fold}/{config.n_splits}: train={len(train_idx)} val={len(val_idx)}")
        result = train_one_fold(
            model_factory, X[train_idx], y[train_idx], X[val_idx], y[val_idx], config
        )
        row = dict(fold=fold, n_train=len(train_idx), n_val=len(val_idx),
                   best_val_loss=result["best_val_loss"], epochs_run=result["epochs_run"])
        for j, name in enumerate(target_names):
            row[f"r2_{name}"] = float(r2_score(result["y_val"][:, j], result["y_val_pred"][:, j]))
            row[f"rmse_{name}"] = float(
                np.sqrt(mean_squared_error(result["y_val"][:, j], result["y_val_pred"][:, j]))
            )
            row[f"mae_{name}"] = float(mean_absolute_error(result["y_val"][:, j], result["y_val_pred"][:, j]))
        row["r2_mean"] = float(np.mean([row[f"r2_{n}"] for n in target_names]))
        row["rmse_mean"] = float(np.mean([row[f"rmse_{n}"] for n in target_names]))
        fold_rows.append(row)
        predictions.append(dict(fold=fold, y_val=result["y_val"], y_val_pred=result["y_val_pred"]))

    folds_df = pd.DataFrame(fold_rows)
    aggregate = {
        "n_folds": len(folds_df),
        "r2_mean": float(folds_df["r2_mean"].mean()),
        "r2_std": float(folds_df["r2_mean"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
        "rmse_mean": float(folds_df["rmse_mean"].mean()),
    }
    for name in target_names:
        aggregate[f"r2_{name}"] = float(folds_df[f"r2_{name}"].mean())
        aggregate[f"rmse_{name}"] = float(folds_df[f"rmse_{name}"].mean())
    return dict(folds=folds_df, aggregate=aggregate, predictions=predictions, config=config.as_dict())


def load_surrogate_dataset(path) -> tuple:
    """Load the ``(r/R, H, DT, H0, Ug) -> (eps_s, C/C0)`` surrogate dataset.

    The dataset itself is **not** part of this repository: it belongs to the
    supervising group and consists of two bed diameters (76 / 152 mm) from
    coupled CFD/experimental campaigns.  Point ``path`` at your own copy.
    """
    df = pd.read_excel(path)
    features = ["r/R", "H", "DT", "H0", "Ug"]
    targets = ["εs", "C/C0"]
    missing = [c for c in features + targets if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    return df[features].to_numpy(dtype=float), df[targets].to_numpy(dtype=float), features, targets

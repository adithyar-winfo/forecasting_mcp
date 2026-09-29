"""Model utilities: errors, metrics, the LightGBM wrapper and an in-memory model cache."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger("forecasting_mcp.models")


class ForecastError(Exception):
    error_type = "forecast_error"


class InvalidRequestError(ForecastError):
    error_type = "invalid_request"


class InsufficientDataError(ForecastError):
    error_type = "insufficient_data"


class DataSourceError(ForecastError):
    error_type = "database_error"


class ModelTrainingError(ForecastError):
    error_type = "model_training_error"


def compute_metrics(actual: Any, predicted: Any) -> dict[str, float | int | None]:
    """MAE, RMSE, WAPE and bias; bias = (sum(pred) - sum(actual)) / sum(actual), positive = over-forecast."""
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    valid = ~(np.isnan(a) | np.isnan(p))
    a, p = a[valid], p[valid]
    if a.size == 0:
        return {"mae": None, "rmse": None, "wape": None, "bias": None, "n": 0}
    err = p - a
    denom = float(np.abs(a).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wape": float(np.abs(err).sum() / denom) if denom > 0 else None,
        "bias": float(err.sum() / denom) if denom > 0 else None,
        "n": int(a.size),
    }


def metric_or_inf(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    return float("inf") if value is None else float(value)


DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
}


class LightGBMModel:
    """Thin wrapper so the forecasting algorithm can be swapped without touching the DB/MCP layers."""

    name = "lightgbm"

    def __init__(self, objective: str = "regression", params: dict[str, Any] | None = None,
                 max_rounds: int = 2000, early_stopping_rounds: int = 100) -> None:
        self.params = {**DEFAULT_LGB_PARAMS, **(params or {}), "objective": objective}
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.booster: lgb.Booster | None = None
        self.best_iteration: int | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame | None = None,
            y_val: np.ndarray | None = None, num_rounds: int | None = None) -> "LightGBMModel":
        if len(X) == 0:
            raise ModelTrainingError("No training rows available.")
        try:
            train = lgb.Dataset(X, label=y, free_raw_data=False)
            if X_val is not None and len(X_val) > 0:
                valid = lgb.Dataset(X_val, label=y_val, reference=train)
                self.booster = lgb.train(
                    self.params, train, num_boost_round=self.max_rounds, valid_sets=[valid],
                    callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
                )
                self.best_iteration = self.booster.best_iteration or self.max_rounds
            else:
                rounds = num_rounds or self.best_iteration or 500
                self.booster = lgb.train(self.params, train, num_boost_round=rounds)
                self.best_iteration = rounds
        except ModelTrainingError:
            raise
        except Exception as exc:
            raise ModelTrainingError(f"LightGBM training failed: {exc}") from exc
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise ModelTrainingError("Model has not been trained.")
        return self.booster.predict(X, num_iteration=self.best_iteration)

    def feature_importance(self, top: int = 10) -> dict[str, float]:
        if self.booster is None:
            return {}
        gains = self.booster.feature_importance(importance_type="gain")
        ranked = sorted(zip(self.booster.feature_name(), gains), key=lambda kv: kv[1], reverse=True)
        return {name: round(float(gain), 2) for name, gain in ranked[:top]}


@dataclass
class ModelArtifact:
    key: str
    model_name: str
    version: str
    trained_at: str
    data_signature: tuple
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def make_version(model_name: str, data_date: Any) -> tuple[str, str]:
    trained = datetime.now(timezone.utc)
    return f"{model_name}-{pd.Timestamp(data_date):%Y%m%d}-{trained:%Y%m%d%H%M%S}", trained.isoformat(timespec="seconds")


class ModelCache:
    """In-memory artifact cache; an artifact is reused while its source-data signature is unchanged."""

    def __init__(self) -> None:
        self._items: dict[str, ModelArtifact] = {}
        self._lock = threading.Lock()

    def get_or_train(self, key: str, signature: tuple, train_fn: Callable[[], ModelArtifact],
                     force: bool = False) -> tuple[ModelArtifact, bool]:
        with self._lock:
            cached = self._items.get(key)
            if cached is not None and not force and cached.data_signature == signature:
                return cached, True
            logger.info("Training '%s' model (signature=%s, force=%s)", key, signature, force)
            artifact = train_fn()
            self._items[key] = artifact
            return artifact, False

"""Implementações concretas de classificadores de crédito (XGBoost e Sklearn)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from credit_guard.data.preprocessor import FEATURE_COLUMNS, build_preprocessing_pipeline
from credit_guard.model.base import BaseClassifier

logger = logging.getLogger(__name__)


def _build_gradient_boosting_estimator(
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
) -> tuple[Any, str]:
    """Instancia XGBClassifier ou HistGradientBoostingClassifier se libomp ausente."""
    try:
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state,
        )
        return estimator, "xgboost_native"
    except Exception as exc:  # pragma: no cover - fallback para macOS sem libomp
        logger.info(
            "Biblioteca nativa libomp/xgboost indisponível (%s); usando "
            "HistGradientBoostingClassifier (Scikit-Learn) equivalente.",
            type(exc).__name__,
        )
        estimator = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        return estimator, "sklearn_hist_gradient_boosting"


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Calcula métricas financeiras e estatísticas de classificação binária."""
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "roc_auc": round(auc, 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "predicted_default_rate": round(float(np.mean(y_pred)), 4),
        "mean_predicted_probability": round(float(np.mean(y_prob)), 4),
        "actual_default_rate": round(float(np.mean(y_true)), 4),
    }


class XGBoostCreditClassifier(BaseClassifier):
    """Classificador de Credit Scoring baseado em Gradient Boosting."""

    def __init__(
        self,
        n_estimators: int = 140,
        max_depth: int = 4,
        learning_rate: float = 0.07,
        random_state: int = 42,
        threshold: float = 0.40,
    ) -> None:
        """Inicializa o pipeline de pré-processamento acoplado ao Gradient Boosting."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.threshold = threshold
        estimator, backend_name = _build_gradient_boosting_estimator(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.backend_engine = backend_name
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessing_pipeline()),
                ("classifier", estimator),
            ]
        )

    @property
    def model_name(self) -> str:
        """Retorna o identificador do classificador XGBoost."""
        return "xgboost"

    def get_params(self) -> dict[str, Any]:
        """Retorna os hiperparâmetros principais para registro no MLflow."""
        return {
            "model_type": self.model_name,
            "backend_engine": self.backend_engine,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "decision_threshold": self.threshold,
            "random_state": self.random_state,
        }

    def fit(self, x: pd.DataFrame, y: pd.Series) -> XGBoostCreditClassifier:
        """Treina o pipeline completo sobre as colunas preditoras de referência."""
        self.pipeline.fit(x[FEATURE_COLUMNS], y.astype(int))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Estima a probabilidade de inadimplência (`default_risk = 1`)."""
        probs = self.pipeline.predict_proba(x[FEATURE_COLUMNS])[:, 1]
        return np.asarray(probs, dtype=float)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Classifica propostas usando o limiar de corte configurado."""
        probs = self.predict_proba(x)
        return (probs >= self.threshold).astype(int)

    def evaluate(self, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Avalia o desempenho preditivo e calibração do modelo."""
        y_true = y.to_numpy(dtype=int)
        y_prob = self.predict_proba(x)
        y_pred = (y_prob >= self.threshold).astype(int)
        return _compute_classification_metrics(y_true, y_pred, y_prob)

    def save(self, path: Path) -> Path:
        """Salva a instância serializada com joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> XGBoostCreditClassifier:
        """Carrega instância serializada do disco."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            msg = f"Objeto carregado de {path} não é {cls.__name__}"
            raise TypeError(msg)
        return loaded


class LogisticCreditClassifier(BaseClassifier):
    """Classificador linear regularizado (Scikit-Learn LogisticRegression)."""

    def __init__(
        self,
        c_reg: float = 1.0,
        max_iter: int = 500,
        random_state: int = 42,
        threshold: float = 0.40,
    ) -> None:
        """Inicializa o pipeline de regressão logística com pesos balanceados."""
        self.c_reg = c_reg
        self.max_iter = max_iter
        self.random_state = random_state
        self.threshold = threshold
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessing_pipeline()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_reg,
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    @property
    def model_name(self) -> str:
        """Retorna o identificador do classificador logístico."""
        return "logistic"

    def get_params(self) -> dict[str, Any]:
        """Retorna os hiperparâmetros principais para rastreamento."""
        return {
            "model_type": self.model_name,
            "C": self.c_reg,
            "max_iter": self.max_iter,
            "decision_threshold": self.threshold,
            "random_state": self.random_state,
        }

    def fit(self, x: pd.DataFrame, y: pd.Series) -> LogisticCreditClassifier:
        """Ajusta o modelo logístico nos dados de referência."""
        self.pipeline.fit(x[FEATURE_COLUMNS], y.astype(int))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Estima a probabilidade de default (`default_risk = 1`)."""
        probs = self.pipeline.predict_proba(x[FEATURE_COLUMNS])[:, 1]
        return np.asarray(probs, dtype=float)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Retorna a decisão binária de risco."""
        probs = self.predict_proba(x)
        return (probs >= self.threshold).astype(int)

    def evaluate(self, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Calcula métricas de classificação para o modelo logístico."""
        y_true = y.to_numpy(dtype=int)
        y_prob = self.predict_proba(x)
        y_pred = (y_prob >= self.threshold).astype(int)
        return _compute_classification_metrics(y_true, y_pred, y_prob)

    def save(self, path: Path) -> Path:
        """Serializa o classificador logístico em disco."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> LogisticCreditClassifier:
        """Carrega instância serializada do disco."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            msg = f"Objeto carregado de {path} não é {cls.__name__}"
            raise TypeError(msg)
        return loaded

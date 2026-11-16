"""Classe abstrata definindo o contrato para classificadores de risco de crédito."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseClassifier(ABC):
    """Contrato abstrato para todos os classificadores de Credit Scoring."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Retorna o identificador único da implementação do modelo."""

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> BaseClassifier:
        """Treina o pipeline de pré-processamento e o classificador supervisionado."""

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Retorna predições binárias de default (0 = aprovado, 1 = alto risco)."""

    @abstractmethod
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades estimadas da classe positiva (`default_risk=1`)."""

    @abstractmethod
    def evaluate(self, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Calcula métricas de qualidade preditiva e calibração na carteira avaliada."""

    @abstractmethod
    def save(self, path: Path) -> Path:
        """Serializa o modelo treinado em disco."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> BaseClassifier:
        """Carrega uma instância treinada do modelo a partir do disco."""

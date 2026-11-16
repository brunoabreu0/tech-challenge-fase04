"""Padrão Factory para instanciação e carregamento de classificadores de crédito."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from credit_guard.data.contracts import DataContractValidator
from credit_guard.data.loader import (
    load_or_create_reference_dataset,
    split_features_and_target,
)
from credit_guard.model.base import BaseClassifier
from credit_guard.model.classifier import (
    LogisticCreditClassifier,
    XGBoostCreditClassifier,
)
from credit_guard.settings import get_settings

_REGISTRY: dict[str, type[BaseClassifier]] = {
    "xgboost": XGBoostCreditClassifier,
    "logistic": LogisticCreditClassifier,
}


class ClassifierFactory:
    """Fábrica de modelos seguindo o contrato `BaseClassifier`."""

    @staticmethod
    def create(model_type: str = "xgboost", **kwargs: Any) -> BaseClassifier:
        """Instancia um classificador pelo identificador (`xgboost` ou `logistic`)."""
        key = model_type.strip().lower()
        if key not in _REGISTRY:
            supported = ", ".join(sorted(_REGISTRY.keys()))
            msg = f"Modelo '{model_type}' desconhecido. Opções válidas: {supported}"
            raise ValueError(msg)
        return _REGISTRY[key](**kwargs)

    @staticmethod
    def load(model_type: str, path: Path) -> BaseClassifier:
        """Carrega um classificador previamente treinado do caminho informado."""
        key = model_type.strip().lower()
        if key not in _REGISTRY:
            supported = ", ".join(sorted(_REGISTRY.keys()))
            msg = f"Modelo '{model_type}' desconhecido. Opções válidas: {supported}"
            raise ValueError(msg)
        return _REGISTRY[key].load(path)

    @classmethod
    def load_or_train_default(
        cls,
        model_dir: Path | None = None,
        model_type: str | None = None,
    ) -> BaseClassifier:
        """Carrega o modelo salvo ou treina automaticamente sobre a base validada."""
        settings = get_settings()
        target_dir = model_dir or settings.model_dir
        chosen_type = (model_type or settings.default_model_type).strip().lower()
        model_path = target_dir / f"credit_{chosen_type}.joblib"

        if model_path.exists():
            return cls.load(chosen_type, model_path)

        df = load_or_create_reference_dataset()
        validator = DataContractValidator(require_target=True)
        validation = validator.validate_batch(df)
        if not validation.is_valid:
            msg = (
                f"Treinamento abortado: Dataset de Referência violou contrato Pandera "
                f"({validation.violation_count} falhas)."
            )
            raise ValueError(msg)

        x, y = split_features_and_target(df)
        clf = cls.create(chosen_type, random_state=settings.random_seed)
        clf.fit(x, y)
        clf.save(model_path)
        return clf

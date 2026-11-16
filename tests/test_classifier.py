"""Testes unitários para classificadores de crédito e ClassifierFactory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_guard.data.loader import split_features_and_target
from credit_guard.model.classifier import (
    LogisticCreditClassifier,
    XGBoostCreditClassifier,
)
from credit_guard.model.factory import ClassifierFactory


def test_xgboost_and_logistic_classifiers_fit_predict_save_load(
    reference_credit_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Testa treino, predição, avaliação e persistência do XGBoost e Logistic."""
    x, y = split_features_and_target(reference_credit_df)

    for model_type, expected_cls in (
        ("xgboost", XGBoostCreditClassifier),
        ("logistic", LogisticCreditClassifier),
    ):
        clf = ClassifierFactory.create(model_type, random_state=42)
        assert isinstance(clf, expected_cls)
        assert clf.model_name == model_type
        assert "model_type" in clf.get_params()

        clf.fit(x, y)
        probs = clf.predict_proba(x.head(25))
        preds = clf.predict(x.head(25))
        assert len(probs) == 25
        assert len(preds) == 25
        assert ((probs >= 0.0) & (probs <= 1.0)).all()
        assert set(preds).issubset({0, 1})

        metrics = clf.evaluate(x, y)
        assert metrics["roc_auc"] > 0.65
        assert 0.0 <= metrics["brier_score"] <= 1.0

        saved_path = clf.save(tmp_path / f"{model_type}.joblib")
        loaded = ClassifierFactory.load(model_type, saved_path)
        assert loaded.model_name == model_type


def test_factory_invalid_model_and_type_error(
    reference_credit_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifica erros apropriados ao solicitar modelo inválido ou tipo incompatível."""
    with pytest.raises(ValueError, match="desconhecido"):
        ClassifierFactory.create("modelo_inexistente")

    with pytest.raises(ValueError, match="desconhecido"):
        ClassifierFactory.load("modelo_inexistente", tmp_path / "dummy.joblib")

    x, y = split_features_and_target(reference_credit_df)
    log_clf = LogisticCreditClassifier().fit(x, y)
    p = log_clf.save(tmp_path / "wrong.joblib")
    with pytest.raises(TypeError):
        XGBoostCreditClassifier.load(p)


def test_load_or_train_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Testa o fallback automático de treino e subsequente carregamento do modelo."""
    monkeypatch.setenv("REFERENCE_SAMPLES", "5000")
    clf1 = ClassifierFactory.load_or_train_default(
        model_dir=tmp_path, model_type="logistic"
    )
    assert (tmp_path / "credit_logistic.joblib").exists()

    clf2 = ClassifierFactory.load_or_train_default(
        model_dir=tmp_path, model_type="logistic"
    )
    assert clf1.model_name == clf2.model_name == "logistic"

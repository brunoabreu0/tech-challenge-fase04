"""Testes unitários para configurações, carregamento de dados e camada LGPD."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_guard.data.loader import (
    _normalize_kaggle_credit_df,
    compute_true_default_probability,
    download_kaggle_dataset,
    generate_credit_dataset,
    load_or_create_reference_dataset,
    split_features_and_target,
)
from credit_guard.data.preprocessor import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
)
from credit_guard.data.privacy import DIRECT_PII_COLUMNS, PrivacyGuard
from credit_guard.settings import get_settings


def test_settings_defaults() -> None:
    """Garante que as configurações padrão atendam aos requisitos (>= 5000 amostras)."""
    settings = get_settings()
    assert settings.reference_samples >= 5000
    assert settings.psi_warning_threshold == 0.10
    assert settings.psi_critical_threshold == 0.25


def test_generate_credit_dataset_exceeds_5000_samples() -> None:
    """Valida a geração determinística do dataset de crédito com >= 5.000 amostras."""
    df = generate_credit_dataset(n_samples=5200, seed=42, include_pii=True)
    assert len(df) == 5200
    for col in FEATURE_COLUMNS + [TARGET_COLUMN, "cpf", "full_name", "zip_code"]:
        assert col in df.columns
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_privacy_guard_sanitizes_pii_and_audits(raw_credit_df: pd.DataFrame) -> None:
    """Testa pseudonimização, mascaramento de CPF, CEP e relatório LGPD."""
    guard = PrivacyGuard(salt="test-salt", retention_days=1825)
    assert guard.mask_cpf("123.456.789-01") == "***.456.789-**"
    assert guard.mask_cpf("invalido") == "***.***.***-**"
    assert guard.generalize_zip_code("01310-100") == "013XX-XXX"
    assert guard.generalize_zip_code("12") == "000XX-XXX"

    sanitized = guard.sanitize_dataframe(raw_credit_df, drop_direct_pii=True)
    for pii_col in DIRECT_PII_COLUMNS:
        assert pii_col not in sanitized.columns
    assert "customer_token" in sanitized.columns
    assert sanitized["zip_code"].iloc[0].endswith("XX-XXX")

    report = guard.generate_privacy_audit_report(raw_credit_df, sanitized)
    assert report["is_compliant"] is True
    assert report["data_minimization"]["remaining_direct_pii_in_model_input"] == []
    assert "disparate_impact_ratio" in report["fairness_and_non_discrimination"]


def test_load_or_create_reference_dataset_and_preprocessor(tmp_path: Path) -> None:
    """Testa persistência/leitura de CSV e transformação sem data leakage."""
    df1 = load_or_create_reference_dataset(
        raw_dir=tmp_path, n_samples=5000, seed=42, sanitize_pii=True
    )
    assert (tmp_path / "credit_reference.csv").exists()
    assert len(df1) >= 5000
    assert "cpf" not in df1.columns

    # Segunda chamada lê do CSV já existente
    df2 = load_or_create_reference_dataset(
        raw_dir=tmp_path, n_samples=5000, seed=42, sanitize_pii=True
    )
    assert len(df2) == len(df1)

    x, y = split_features_and_target(df1)
    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(x)
    assert transformed.shape[0] == len(df1)
    assert len(y) == len(df1)


def test_compute_true_default_probability_concept_drift(
    reference_credit_df: pd.DataFrame,
) -> None:
    """Verifica que o choque de concept drift eleva o risco médio esperado."""
    p_base = compute_true_default_probability(reference_credit_df)
    p_drift = compute_true_default_probability(
        reference_credit_df, macro_stress_bias=0.3, concept_drift_strength=1.0
    )
    assert float(p_drift.mean()) > float(p_base.mean())


def test_normalize_kaggle_credit_df_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Testa normalização de dados Kaggle e fallback seguro quando sem credenciais."""
    sample_kaggle = pd.DataFrame(
        {
            "person_age": [25, 34, 45],
            "person_income": [48000, 82000, 120000],
            "person_home_ownership": ["RENT", "MORTGAGE", "OWN"],
            "person_emp_length": [3.0, 8.0, 15.0],
            "loan_intent": ["EDUCATION", "DEBTCONSOLIDATION", "VENTURE"],
            "loan_amnt": [10000, 18000, 25000],
            "loan_int_rate": [11.5, 13.2, 9.8],
            "loan_percent_income": [0.21, 0.22, 0.21],
        }
    )
    norm_df = _normalize_kaggle_credit_df(sample_kaggle, seed=42)
    assert len(norm_df) == 3
    for col in FEATURE_COLUMNS + [TARGET_COLUMN, "cpf"]:
        assert col in norm_df.columns

    monkeypatch.setattr(
        "credit_guard.data.loader._resolve_kaggle_credentials", lambda: False
    )
    assert download_kaggle_dataset(tmp_path) is False

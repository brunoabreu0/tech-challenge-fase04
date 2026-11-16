"""Gerador e carregador determinístico do Dataset de Crédito (>= 5.000 amostras)."""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from credit_guard.data.preprocessor import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    VALID_HOME_OWNERSHIP,
    VALID_LOAN_PURPOSES,
)
from credit_guard.data.privacy import PrivacyGuard
from credit_guard.settings import get_settings

logger = logging.getLogger(__name__)

_KAGGLE_DATASET = "laotse/credit-risk-dataset"


def _resolve_kaggle_credentials() -> bool:
    """Resolve credenciais do Kaggle via KAGGLE_API_TOKEN ou ~/.kaggle/kaggle.json."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    api_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if api_token:
        try:
            kaggle_json.parent.mkdir(parents=True, exist_ok=True)
            kaggle_json.write_text(json.dumps({"token": api_token}), encoding="utf-8")
            kaggle_json.chmod(0o600)
            return True
        except OSError as exc:
            logger.warning("Falha ao gravar ~/.kaggle/kaggle.json: %s", exc)

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    return kaggle_json.exists()


def _normalize_kaggle_credit_df(
    raw_kaggle: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """Normaliza o dataset `laotse/credit-risk-dataset` para o contrato Pandera."""
    df_k = raw_kaggle.dropna().copy()
    df_k = df_k[(df_k["person_age"] > 18) & (df_k["person_age"] <= 95)]
    df_k = df_k[(df_k["person_income"] >= 1000) & (df_k["person_income"] <= 5_000_000)]
    df_k = df_k[df_k["person_emp_length"] >= 0]
    df_k = df_k[df_k["person_emp_length"] <= (df_k["person_age"] - 18)].reset_index(
        drop=True
    )

    n = len(df_k)
    rng = np.random.default_rng(seed)

    intent_map = {
        "DEBTCONSOLIDATION": "DEBT_CONSOLIDATION",
        "HOMEIMPROVEMENT": "HOME_IMPROVEMENT",
        "VENTURE": "SMALL_BUSINESS",
        "EDUCATION": "EDUCATION",
        "PERSONAL": "PERSONAL",
        "MEDICAL": "CREDIT_CARD",
    }
    ownership_map = {
        "OWN": "OWN",
        "MORTGAGE": "MORTGAGE",
        "RENT": "RENT",
        "OTHER": "OTHER",
    }

    base = generate_credit_dataset(n_samples=n, seed=seed, include_pii=True)
    base["age"] = df_k["person_age"].astype(int).to_numpy()
    base["annual_income"] = np.round(df_k["person_income"].astype(float).to_numpy(), 2)
    base["employment_length_years"] = np.round(
        df_k["person_emp_length"].astype(float).to_numpy(), 1
    )
    base["loan_amount"] = np.round(
        np.clip(df_k["loan_amnt"].astype(float).to_numpy(), 500.0, 500_000.0), 2
    )
    base["interest_rate"] = np.round(
        np.clip(df_k["loan_int_rate"].astype(float).to_numpy(), 1.0, 99.0), 2
    )
    base["debt_to_income_ratio"] = np.round(
        np.clip(df_k["loan_percent_income"].astype(float).to_numpy(), 0.01, 0.98), 4
    )
    base["home_ownership"] = (
        df_k["person_home_ownership"].astype(str).map(ownership_map).fillna("RENT")
    ).to_numpy()
    base["loan_purpose"] = (
        df_k["loan_intent"].astype(str).map(intent_map).fillna("PERSONAL")
    ).to_numpy()
    probs = compute_true_default_probability(base)
    base[TARGET_COLUMN] = (rng.uniform(0.0, 1.0, size=n) < probs).astype(int)
    return base


def download_kaggle_dataset(data_dir: Path, seed: int = 42) -> bool:
    """Tenta baixar o Credit Risk Dataset do Kaggle (`laotse/credit-risk-dataset`)."""
    if not _resolve_kaggle_credentials():
        return False

    try:
        import kaggle  # type: ignore[import-untyped]
    except ImportError:
        logger.info("Pacote 'kaggle' não instalado; usando gerador determinístico.")
        return False

    try:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            _KAGGLE_DATASET,
            path=str(data_dir),
            unzip=False,
            quiet=True,
        )
        zip_path = data_dir / "credit-risk-dataset.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
            zip_path.unlink()

        raw_csv = data_dir / "credit_risk_dataset.csv"
        if raw_csv.exists():
            raw_df = pd.read_csv(raw_csv)
            normalized = _normalize_kaggle_credit_df(raw_df, seed=seed)
            normalized.to_csv(data_dir / "credit_reference.csv", index=False)
            logger.info(
                "Dataset Kaggle normalizado e salvo (%d linhas).", len(normalized)
            )
            return True
        return False
    except Exception as exc:
        logger.warning("Download do Kaggle indisponível (%s); usando fallback.", exc)
        return False


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Função logística numericamente estável para probabilidades de default."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def compute_true_default_probability(
    df: pd.DataFrame,
    macro_stress_bias: float = 0.0,
    concept_drift_strength: float = 0.0,
) -> np.ndarray:
    """Calcula a probabilidade real de inadimplência segundo a equação estrutural.

    Permite avaliar tanto o cenário estacionário (Referência) quanto o cenário
    com Concept Drift (mudança na relação causal entre endividamento, juros e default).
    """
    dti = df["debt_to_income_ratio"].to_numpy(dtype=float)
    utilization = df["credit_utilization"].to_numpy(dtype=float)
    credit_score = df["credit_score"].to_numpy(dtype=float)
    delinq = df["num_delinquencies_2yrs"].to_numpy(dtype=float)
    unemp = df["unemployment_rate_region"].to_numpy(dtype=float)
    income = np.maximum(df["annual_income"].to_numpy(dtype=float), 1000.0)
    loan_amt = df["loan_amount"].to_numpy(dtype=float)
    interest = df["interest_rate"].to_numpy(dtype=float)
    emp_len = df["employment_length_years"].to_numpy(dtype=float)

    loan_to_income = loan_amt / income

    # Equação estrutural base de risco de crédito
    logit = (
        -2.35
        + 2.80 * (dti - 0.30)
        + 2.10 * (utilization - 0.35)
        - 0.0065 * (credit_score - 650.0)
        + 0.48 * delinq
        + 0.095 * (unemp - 7.5)
        + 1.35 * (loan_to_income - 0.25)
        + 0.045 * (interest - 14.0)
        - 0.035 * np.minimum(emp_len, 15.0)
        + macro_stress_bias
    )

    # Concept Drift: sob estresse econômico (inflação alta), clientes com DTI moderado
    # (> 0.35) e uso de cartão (> 0.45) passam a inadimplir muito mais que no passado,
    # mesmo quando possuem credit_score historicamente bom.
    if concept_drift_strength > 0.0:
        nonlinear_shock = concept_drift_strength * (
            3.2 * np.maximum(dti - 0.32, 0.0)
            + 2.6 * np.maximum(utilization - 0.40, 0.0)
            + 0.08 * np.maximum(interest - 15.0, 0.0)
            + 0.12 * np.maximum(unemp - 8.0, 0.0)
        )
        logit = logit + nonlinear_shock

    return _sigmoid(logit)


def generate_credit_dataset(
    n_samples: int = 6000,
    seed: int = 42,
    include_pii: bool = True,
) -> pd.DataFrame:
    """Gera o Dataset de Referência de concessão de crédito (>= 5.000 amostras).

    Reproduz as distribuições estatísticas de carteiras reais de crédito pessoal
    (inspirado em Lending Club / Give Me Some Credit / German Credit).
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(19, 78, size=n_samples)
    # Renda anual em R$ (distribuição log-normal realista entre ~18k e ~450k)
    annual_income = np.round(
        np.clip(
            rng.lognormal(mean=11.05, sigma=0.52, size=n_samples), 15000.0, 600000.0
        ),
        2,
    )
    max_exp = np.maximum(age - 18, 0)
    employment_length_years = np.round(
        np.minimum(rng.exponential(scale=5.5, size=n_samples), max_exp),
        1,
    )
    debt_to_income_ratio = np.round(
        np.clip(rng.beta(a=2.4, b=5.2, size=n_samples), 0.02, 0.92),
        4,
    )
    credit_utilization = np.round(
        np.clip(rng.beta(a=2.2, b=3.8, size=n_samples), 0.01, 0.98),
        4,
    )
    loan_amount = np.round(
        np.clip(
            annual_income * rng.uniform(0.08, 0.48, size=n_samples),
            2000.0,
            180000.0,
        ),
        2,
    )
    credit_score = np.clip(
        rng.normal(loc=665.0, scale=72.0, size=n_samples)
        - 45.0 * debt_to_income_ratio
        - 35.0 * credit_utilization,
        300.0,
        850.0,
    ).astype(int)

    interest_rate = np.round(
        np.clip(
            32.0 - 0.026 * (credit_score - 300) + rng.normal(0.0, 2.1, size=n_samples),
            6.5,
            44.9,
        ),
        2,
    )
    num_delinquencies_2yrs = rng.poisson(
        lam=np.clip(0.25 + 0.85 * credit_utilization, 0.05, 3.0),
        size=n_samples,
    )
    num_delinquencies_2yrs = np.clip(num_delinquencies_2yrs, 0, 12)

    unemployment_rate_region = np.round(
        np.clip(rng.normal(loc=7.4, scale=1.6, size=n_samples), 3.2, 18.0),
        2,
    )

    home_ownership = rng.choice(
        VALID_HOME_OWNERSHIP,
        size=n_samples,
        p=[0.24, 0.36, 0.35, 0.05],
    )
    loan_purpose = rng.choice(
        VALID_LOAN_PURPOSES,
        size=n_samples,
        p=[0.38, 0.24, 0.10, 0.11, 0.06, 0.11],
    )

    data: dict[str, object] = {
        "application_id": [f"APP-{seed}-{i:06d}" for i in range(1, n_samples + 1)],
    }

    if include_pii:
        cpf_nums = rng.integers(100_000_000_00, 999_999_999_99, size=n_samples)
        data["cpf"] = [
            f"{n // 100000000:03d}.{(n // 100000) % 1000:03d}."
            f"{(n // 100) % 1000:03d}-{n % 100:02d}"
            for n in cpf_nums
        ]
        first_names = [
            "Ana",
            "Bruno",
            "Carlos",
            "Daniela",
            "Eduardo",
            "Fernanda",
            "Gabriel",
            "Helena",
        ]
        last_names = [
            "Silva",
            "Santos",
            "Oliveira",
            "Souza",
            "Pereira",
            "Costa",
            "Rodrigues",
            "Almeida",
        ]
        data["full_name"] = [
            f"{rng.choice(first_names)} {rng.choice(last_names)}"
            for _ in range(n_samples)
        ]
        zip_prefixes = rng.integers(1000, 99999, size=n_samples)
        data["zip_code"] = [f"{z:05d}-000" for z in zip_prefixes]

    data.update(
        {
            "age": age,
            "annual_income": annual_income,
            "employment_length_years": employment_length_years,
            "debt_to_income_ratio": debt_to_income_ratio,
            "credit_utilization": credit_utilization,
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "num_delinquencies_2yrs": num_delinquencies_2yrs,
            "unemployment_rate_region": unemployment_rate_region,
            "credit_score": credit_score,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
        }
    )

    df = pd.DataFrame(data)
    probs = compute_true_default_probability(df)
    df[TARGET_COLUMN] = (rng.uniform(0.0, 1.0, size=n_samples) < probs).astype(int)
    return df


def load_or_create_reference_dataset(
    raw_dir: Path | None = None,
    n_samples: int | None = None,
    seed: int | None = None,
    sanitize_pii: bool = True,
) -> pd.DataFrame:
    """Carrega o dataset de referência do disco ou gera deterministicamente.

    Por padrão (`sanitize_pii=True`), aplica o `PrivacyGuard` (Privacy by Default)
    antes de retornar os dados para validação de contratos e treinamento de modelos.
    """
    settings = get_settings()
    target_dir = raw_dir or settings.data_raw_dir
    samples = n_samples or settings.reference_samples
    random_seed = seed if seed is not None else settings.random_seed

    csv_path = target_dir / "credit_reference.csv"
    if not csv_path.exists():
        download_kaggle_dataset(target_dir, seed=random_seed)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = generate_credit_dataset(
            n_samples=samples,
            seed=random_seed,
            include_pii=True,
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    if sanitize_pii:
        guard = PrivacyGuard()
        df = guard.sanitize_dataframe(df, drop_direct_pii=True)

    return df


def split_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as colunas preditoras (`FEATURE_COLUMNS`) do alvo (`default_risk`)."""
    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    return x, y

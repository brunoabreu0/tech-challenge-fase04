"""Definição das features e pipeline de pré-processamento para risco de crédito."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES: list[str] = [
    "age",
    "annual_income",
    "employment_length_years",
    "debt_to_income_ratio",
    "credit_utilization",
    "loan_amount",
    "interest_rate",
    "num_delinquencies_2yrs",
    "unemployment_rate_region",
    "credit_score",
]

CATEGORICAL_FEATURES: list[str] = [
    "home_ownership",
    "loan_purpose",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "default_risk"

VALID_HOME_OWNERSHIP: tuple[str, ...] = ("OWN", "MORTGAGE", "RENT", "OTHER")
VALID_LOAN_PURPOSES: tuple[str, ...] = (
    "DEBT_CONSOLIDATION",
    "CREDIT_CARD",
    "SMALL_BUSINESS",
    "HOME_IMPROVEMENT",
    "EDUCATION",
    "PERSONAL",
)


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Constrói o ColumnTransformer padronizando numéricas e codificando categóricas.

    Garante ausência total de data leakage entre os datasets de Referência e Produção.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

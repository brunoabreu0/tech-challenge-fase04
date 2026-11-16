"""Simulador de Data Drift (Covariate Shift), Prior Shift e Concept Drift em Crédito."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from credit_guard.data.loader import (
    compute_true_default_probability,
    generate_credit_dataset,
)
from credit_guard.data.preprocessor import TARGET_COLUMN
from credit_guard.data.privacy import PrivacyGuard
from credit_guard.model.base import BaseClassifier
from credit_guard.settings import get_settings


@dataclass(frozen=True)
class DriftScenario:
    """Parâmetros de estresse macroeconômico para simulação de Drift em Produção."""

    name: str
    description: str
    income_multiplier: float
    unemployment_shift: float
    dti_shift: float
    utilization_shift: float
    interest_rate_shift: float
    credit_score_shift: int
    macro_stress_bias: float
    concept_drift_strength: float


SCENARIOS: dict[str, DriftScenario] = {
    "baseline_stationary": DriftScenario(
        name="baseline_stationary",
        description="Cenário estacionário sem degradação macroeconômica (controle)",
        income_multiplier=1.0,
        unemployment_shift=0.0,
        dti_shift=0.0,
        utilization_shift=0.0,
        interest_rate_shift=0.0,
        credit_score_shift=0,
        macro_stress_bias=0.0,
        concept_drift_strength=0.0,
    ),
    "moderate_inflation": DriftScenario(
        name="moderate_inflation",
        description="Inflação moderada e aperto monetário inicial (Drift Moderado)",
        income_multiplier=0.86,
        unemployment_shift=2.8,
        dti_shift=0.11,
        utilization_shift=0.13,
        interest_rate_shift=4.2,
        credit_score_shift=-28,
        macro_stress_bias=0.22,
        concept_drift_strength=0.55,
    ),
    "severe_stagflation": DriftScenario(
        name="severe_stagflation",
        description=(
            "Choque severo de inflação, alta de juros SELIC, aumento do desemprego "
            "e mudança de perfil de clientes (Data Drift + Concept Drift severo)"
        ),
        income_multiplier=0.72,
        unemployment_shift=5.8,
        dti_shift=0.21,
        utilization_shift=0.23,
        interest_rate_shift=8.4,
        credit_score_shift=-58,
        macro_stress_bias=0.48,
        concept_drift_strength=1.25,
    ),
}


class DriftSimulator:
    """Gera lotes de Produção simulando degradação temporal (Data + Concept Drift)."""

    def __init__(self, seed: int = 1042) -> None:
        """Inicializa o simulador com semente determinística dedicada à produção."""
        self.seed = seed

    def simulate_production_dataset(
        self,
        n_samples: int = 3000,
        scenario_name: str = "severe_stagflation",
        classifier: BaseClassifier | None = None,
        sanitize_pii: bool = True,
    ) -> pd.DataFrame:
        """Cria um Dataset de Produção com deslocamento estatístico e Concept Drift.

        Todas as variáveis alteradas respeitam os limites físicos do Contrato de Dados
        (Pandera), demonstrando na prática a degradação silenciosa: os dados são
        sintaticamente válidos, mas estatisticamente deslocados em relação ao treino.
        """
        if scenario_name not in SCENARIOS:
            valid = ", ".join(sorted(SCENARIOS.keys()))
            msg = f"Cenário '{scenario_name}' inválido. Escolha entre: {valid}"
            raise ValueError(msg)

        scenario = SCENARIOS[scenario_name]
        rng = np.random.default_rng(self.seed)

        df = generate_credit_dataset(
            n_samples=n_samples,
            seed=self.seed,
            include_pii=True,
        )
        df["application_id"] = [
            f"PROD-{scenario.name[:4].upper()}-{i:06d}" for i in range(1, n_samples + 1)
        ]

        # 1. Corrosão real da renda anual (inflação / novos perfis de menor renda)
        df["annual_income"] = np.round(
            np.clip(
                df["annual_income"]
                * scenario.income_multiplier
                * rng.normal(1.0, 0.06, size=n_samples),
                1500.0,
                600000.0,
            ),
            2,
        )

        # 2. Aumento da taxa de desemprego regional
        df["unemployment_rate_region"] = np.round(
            np.clip(
                df["unemployment_rate_region"]
                + scenario.unemployment_shift
                + rng.normal(0.0, 0.7, size=n_samples),
                3.2,
                35.0,
            ),
            2,
        )

        # 3. Elevação do comprometimento de renda (DTI)
        df["debt_to_income_ratio"] = np.round(
            np.clip(
                df["debt_to_income_ratio"]
                + scenario.dti_shift
                + rng.normal(0.0, 0.03, size=n_samples),
                0.02,
                0.98,
            ),
            4,
        )

        # 4. Uso intensivo do limite de cartão de crédito
        df["credit_utilization"] = np.round(
            np.clip(
                df["credit_utilization"]
                + scenario.utilization_shift
                + rng.normal(0.0, 0.04, size=n_samples),
                0.01,
                0.99,
            ),
            4,
        )

        # 5. Aumento da taxa de juros nas novas concessões
        df["interest_rate"] = np.round(
            np.clip(
                df["interest_rate"] + scenario.interest_rate_shift,
                6.5,
                65.0,
            ),
            2,
        )

        # 6. Deterioração do score de crédito médio
        df["credit_score"] = np.clip(
            df["credit_score"] + scenario.credit_score_shift,
            300,
            850,
        ).astype(int)

        # 7. Mudança estrutural na inadimplência real (Concept Drift + Prior Shift)
        true_probs = compute_true_default_probability(
            df,
            macro_stress_bias=scenario.macro_stress_bias,
            concept_drift_strength=scenario.concept_drift_strength,
        )
        df[TARGET_COLUMN] = (rng.uniform(0.0, 1.0, size=n_samples) < true_probs).astype(
            int
        )

        if sanitize_pii:
            guard = PrivacyGuard()
            df = guard.sanitize_dataframe(df, drop_direct_pii=True)

        if classifier is not None:
            df["predicted_probability"] = np.round(classifier.predict_proba(df), 4)
            df["prediction"] = classifier.predict(df).astype(int)

        return df


def load_or_create_production_dataset(
    processed_dir: Path | None = None,
    n_samples: int | None = None,
    scenario_name: str = "severe_stagflation",
    classifier: BaseClassifier | None = None,
) -> pd.DataFrame:
    """Gera e persiste o Dataset de Produção degradado com predições do modelo."""
    settings = get_settings()
    target_dir = processed_dir or settings.data_processed_dir
    samples = n_samples or settings.production_samples

    simulator = DriftSimulator(seed=settings.random_seed + 1000)
    prod_df = simulator.simulate_production_dataset(
        n_samples=samples,
        scenario_name=scenario_name,
        classifier=classifier,
        sanitize_pii=True,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / f"credit_production_{scenario_name}.csv"
    prod_df.to_csv(csv_path, index=False)
    return prod_df

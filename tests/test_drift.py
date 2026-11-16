"""Testes unitários para Etapa 2: Simulação e Detecção de Drift (PSI, KS, Evidently)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_guard.data.loader import split_features_and_target
from credit_guard.drift.detector import (
    DriftDetector,
    compute_psi_numeric,
    render_executive_drift_html,
)
from credit_guard.drift.simulator import (
    DriftSimulator,
    load_or_create_production_dataset,
)
from credit_guard.model.factory import ClassifierFactory


def test_stationary_scenario_has_no_severe_drift(
    reference_credit_df: pd.DataFrame,
) -> None:
    """Verifica que o cenário estacionário de controle não aciona alerta crítico."""
    simulator = DriftSimulator(seed=999)
    stationary_df = simulator.simulate_production_dataset(
        n_samples=600,
        scenario_name="baseline_stationary",
    )
    detector = DriftDetector()
    summary = detector.analyze(
        reference_df=reference_credit_df,
        production_df=stationary_df,
        scenario_name="baseline_stationary",
    )
    assert summary.max_psi < 0.25


def test_severe_stagflation_detects_data_and_concept_drift_with_evidently(
    reference_credit_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Testa detecção de Data Drift + Concept Drift e geração de HTML do Evidently."""
    x, y = split_features_and_target(reference_credit_df)
    clf = ClassifierFactory.create("logistic", random_state=42).fit(x, y)

    prod_df = load_or_create_production_dataset(
        processed_dir=tmp_path,
        n_samples=600,
        scenario_name="severe_stagflation",
        classifier=clf,
    )
    assert "predicted_probability" in prod_df.columns

    detector = DriftDetector()
    summary, evidently_html, summary_json = detector.generate_evidently_reports(
        reference_df=reference_credit_df,
        production_df=prod_df,
        classifier=clf,
        scenario_name="severe_stagflation",
        output_dir=tmp_path,
    )
    exec_html = render_executive_drift_html(summary, tmp_path / "exec.html")

    assert evidently_html.exists()
    assert summary_json.exists()
    assert exec_html.exists()
    assert summary.dataset_drift_detected is True
    assert summary.overall_severity == "CRITICAL"
    assert summary.drifted_features_count >= 5
    for expected_col in (
        "unemployment_rate_region",
        "debt_to_income_ratio",
        "credit_utilization",
        "interest_rate",
        "annual_income",
    ):
        assert expected_col in summary.drifted_feature_names

    assert summary.concept_drift is not None
    assert summary.concept_drift.concept_drift_detected is True
    assert summary.concept_drift.cusum_alert_triggered is True


def test_simulator_invalid_scenario_and_psi_edge_cases() -> None:
    """Testa validação de cenário inválido e cálculo de PSI em casos extremos."""
    sim = DriftSimulator()
    with pytest.raises(ValueError, match="inválido"):
        sim.simulate_production_dataset(n_samples=50, scenario_name="invalido")

    import numpy as np

    assert compute_psi_numeric(np.array([]), np.array([1.0, 2.0])) == 0.0
    assert compute_psi_numeric(np.ones(20), np.ones(20)) == 0.0

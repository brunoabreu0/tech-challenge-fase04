"""Testes unitários para a Etapa 3: Observabilidade de Pipelines e MLflow Tracking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_guard.data.contracts import (
    DataContractValidator,
    generate_corrupted_batch,
)
from credit_guard.data.loader import split_features_and_target
from credit_guard.data.privacy import PrivacyGuard
from credit_guard.drift.detector import DriftDetector
from credit_guard.drift.simulator import DriftSimulator
from credit_guard.model.factory import ClassifierFactory
from credit_guard.observability.logger import PipelineObservabilityTracker
from credit_guard.observability.tracker import MLflowObservabilityTracker


def test_pipeline_observability_tracker_records_events_and_errors(
    tmp_path: Path,
) -> None:
    """Testa gravação JSONL, cálculo de saúde operacional e captura de exceção."""
    tracker = PipelineObservabilityTracker(reports_dir=tmp_path)
    idle_health = tracker.get_health_summary()
    assert idle_health["pipeline_status"] == "IDLE"

    with tracker.track_stage("ingestion", rows_processed=5000) as ctx:
        ctx["details"]["source"] = "reference"

    with tracker.track_stage("contract_gatekeeper", rows_processed=120) as ctx:
        ctx["status"] = "BLOCKED"
        ctx["contract_violations"] = 14

    with tracker.track_stage("drift_check", rows_processed=1000) as ctx:
        ctx["status"] = "DRIFT_ALERT"
        ctx["drift_alerts"] = 6

    health = tracker.get_health_summary()
    assert health["pipeline_status"] == "DRIFT_WARNING"
    assert health["total_stage_executions"] == 3
    assert health["total_rows_processed"] == 6120
    assert health["total_blocked_batches"] == 1
    assert health["total_contract_violations"] == 14
    assert health["total_drift_alerts"] == 6

    with (
        pytest.raises(RuntimeError, match="falha simulada"),
        tracker.track_stage("failing_stage"),
    ):
        raise RuntimeError("falha simulada")

    degraded = tracker.get_health_summary()
    assert degraded["pipeline_status"] == "DEGRADED_ERROR"
    assert degraded["error_count"] == 1


def test_mlflow_observability_tracker_logs_run_and_artifacts(
    raw_credit_df: pd.DataFrame,
    reference_credit_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Testa registro de parâmetros, métricas, modelo e artefatos no MLflow."""
    guard = PrivacyGuard()
    privacy_report = guard.generate_privacy_audit_report(
        raw_credit_df, reference_credit_df
    )
    validator = DataContractValidator(require_target=True)
    clean_res = validator.validate_batch(reference_credit_df)
    corrupted_res = validator.validate_batch(
        generate_corrupted_batch(reference_credit_df, n_samples=40)
    )

    x, y = split_features_and_target(reference_credit_df)
    clf = ClassifierFactory.create("logistic", random_state=42).fit(x, y)

    prod_df = DriftSimulator(seed=123).simulate_production_dataset(
        n_samples=300,
        scenario_name="severe_stagflation",
        classifier=clf,
    )
    detector = DriftDetector()
    drift_summary = detector.analyze(
        reference_df=reference_credit_df,
        production_df=prod_df,
        classifier=clf,
        scenario_name="severe_stagflation",
    )

    dummy_artifact = tmp_path / "dummy_report.json"
    dummy_artifact.write_text('{"ok": true}', encoding="utf-8")

    db_uri = f"sqlite:///{tmp_path / 'test_mlflow.db'}"
    ml_tracker = MLflowObservabilityTracker(
        tracking_uri=db_uri,
        experiment_name="pytest-credit-drift",
    )
    run_id = ml_tracker.log_monitoring_cycle(
        run_name="test-run",
        classifier=clf,
        reference_df=reference_credit_df,
        clean_contract=clean_res,
        corrupted_contract=corrupted_res,
        drift_summary=drift_summary,
        privacy_report=privacy_report,
        artifact_paths=[dummy_artifact],
    )
    assert isinstance(run_id, str)
    assert len(run_id) > 8

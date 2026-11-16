"""Integração com MLflow Tracking e Model Registry para Governança e Observabilidade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from credit_guard.data.contracts import ContractValidationResult
from credit_guard.data.preprocessor import FEATURE_COLUMNS
from credit_guard.drift.detector import DriftAnalysisSummary
from credit_guard.model.base import BaseClassifier
from credit_guard.settings import get_settings


class MLflowObservabilityTracker:
    """Registra experimentos, modelo, métricas de drift e artefatos no MLflow."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        """Configura o URI e o experimento do MLflow."""
        settings = get_settings()
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def log_monitoring_cycle(
        self,
        run_name: str,
        classifier: BaseClassifier,
        reference_df: pd.DataFrame,
        clean_contract: ContractValidationResult,
        corrupted_contract: ContractValidationResult,
        drift_summary: DriftAnalysisSummary,
        privacy_report: dict[str, Any],
        artifact_paths: list[Path] | None = None,
    ) -> str:
        """Registra execução completa de validação, treino, drift e LGPD no MLflow."""
        with mlflow.start_run(run_name=run_name) as run:
            params: dict[str, Any] = {
                "model_name": classifier.model_name,
                "scenario_name": drift_summary.scenario_name,
                "reference_rows": drift_summary.reference_rows,
                "production_rows": drift_summary.production_rows,
                "overall_drift_severity": drift_summary.overall_severity,
                "lgpd_compliant": privacy_report.get("is_compliant", True),
            }
            if hasattr(classifier, "get_params"):
                params.update(classifier.get_params())
            mlflow.log_params(params)

            metrics: dict[str, float] = {
                "contract_reference_valid": 1.0 if clean_contract.is_valid else 0.0,
                "contract_corrupted_blocked": (
                    1.0 if corrupted_contract.blocked_ingestion else 0.0
                ),
                "contract_corrupted_violations": float(
                    corrupted_contract.violation_count
                ),
                "drift_detected_flag": (
                    1.0 if drift_summary.dataset_drift_detected else 0.0
                ),
                "drifted_features_count": float(drift_summary.drifted_features_count),
                "share_of_drifted_features": drift_summary.share_of_drifted_features,
                "max_psi": drift_summary.max_psi,
                "mean_psi": drift_summary.mean_psi,
            }

            if drift_summary.prediction_psi is not None:
                metrics["prediction_psi"] = drift_summary.prediction_psi

            for fm in drift_summary.feature_metrics:
                metrics[f"psi_{fm.feature}"] = fm.psi
                if fm.ks_statistic is not None:
                    metrics[f"ks_stat_{fm.feature}"] = fm.ks_statistic

            if drift_summary.concept_drift is not None:
                cd = drift_summary.concept_drift
                metrics.update(
                    {
                        "reference_roc_auc": cd.reference_roc_auc,
                        "production_roc_auc": cd.production_roc_auc,
                        "roc_auc_drop": cd.roc_auc_drop,
                        "reference_f1": cd.reference_f1,
                        "production_f1": cd.production_f1,
                        "reference_brier": cd.reference_brier,
                        "production_brier": cd.production_brier,
                        "unpredicted_risk_gap": cd.unpredicted_risk_gap,
                        "cusum_max_score": cd.cusum_max_score,
                        "concept_drift_flag": (
                            1.0 if cd.concept_drift_detected else 0.0
                        ),
                    }
                )

            fairness = privacy_report.get("fairness_and_non_discrimination", {})
            if "disparate_impact_ratio" in fairness:
                metrics["lgpd_disparate_impact_ratio"] = float(
                    fairness["disparate_impact_ratio"]
                )

            mlflow.log_metrics(metrics)

            # Loga a assinatura formal do modelo (input/output contract no MLflow)
            sample_x = reference_df[FEATURE_COLUMNS].head(50)
            sample_preds = classifier.predict_proba(sample_x)
            signature = infer_signature(sample_x, sample_preds)
            if hasattr(classifier, "pipeline"):
                mlflow.sklearn.log_model(
                    sk_model=classifier.pipeline,
                    name="credit_scoring_model",
                    signature=signature,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )

            for path in artifact_paths or []:
                if path.exists():
                    mlflow.log_artifact(str(path))

            return str(run.info.run_id)

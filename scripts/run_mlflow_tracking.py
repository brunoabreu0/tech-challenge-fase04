"""Script CLI da Etapa 3: Observabilidade de Pipelines, Logs e Rastreamento MLflow."""

from __future__ import annotations

import json
import logging

from credit_guard.data.contracts import (
    DataContractValidator,
    export_data_contract_specification,
    generate_corrupted_batch,
    save_validation_report_json,
)
from credit_guard.data.loader import (
    generate_credit_dataset,
    load_or_create_reference_dataset,
)
from credit_guard.data.privacy import PrivacyGuard
from credit_guard.drift.detector import DriftDetector, render_executive_drift_html
from credit_guard.drift.simulator import load_or_create_production_dataset
from credit_guard.model.factory import ClassifierFactory
from credit_guard.observability.logger import PipelineObservabilityTracker
from credit_guard.observability.tracker import MLflowObservabilityTracker
from credit_guard.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Executa o pipeline monitorado e registra métricas/artefatos no MLflow."""
    settings = get_settings()
    pipe_logger = PipelineObservabilityTracker()
    guard = PrivacyGuard()
    validator = DataContractValidator(require_target=True)

    # 1. Ingestão e Sanitização LGPD
    with pipe_logger.track_stage("1_ingestion_and_lgpd_privacy") as ctx:
        raw_df = generate_credit_dataset(
            n_samples=settings.reference_samples,
            seed=settings.random_seed,
            include_pii=True,
        )
        ref_df = load_or_create_reference_dataset()
        privacy_report = guard.generate_privacy_audit_report(raw_df, ref_df)
        privacy_path = settings.reports_dir / "lgpd_privacy_audit_report.json"
        privacy_path.parent.mkdir(parents=True, exist_ok=True)
        privacy_path.write_text(
            json.dumps(privacy_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        ctx["rows_processed"] = len(ref_df)
        ctx["details"] = {"lgpd_compliant": privacy_report["is_compliant"]}

    # 2. Validação de Contrato no Dataset Limpo
    with pipe_logger.track_stage("2_reference_contract_validation") as ctx:
        clean_res = validator.validate_batch(ref_df)
        ctx["rows_processed"] = clean_res.total_rows
        ctx["contract_violations"] = clean_res.violation_count

    # 3. Bloqueio de Lote Corrompido pelo Gatekeeper Pandera
    with pipe_logger.track_stage("3_corrupted_batch_gatekeeper_test") as ctx:
        corrupted_df = generate_corrupted_batch(ref_df, n_samples=120, seed=42)
        corrupted_res = validator.validate_batch(corrupted_df)
        ctx["status"] = "BLOCKED" if corrupted_res.blocked_ingestion else "SUCCESS"
        ctx["rows_processed"] = corrupted_res.total_rows
        ctx["contract_violations"] = corrupted_res.violation_count
        ctx["details"] = {"violations_by_column": corrupted_res.violations_by_column}

    spec_path = export_data_contract_specification(
        settings.reports_dir / "data_contract_spec.yaml"
    )
    val_report_path = save_validation_report_json(
        clean_res,
        corrupted_res,
        settings.reports_dir / "data_contract_validation_report.json",
    )

    # 4. Carregamento/Treinamento do Classificador Base
    with pipe_logger.track_stage("4_model_training_and_evaluation") as ctx:
        clf = ClassifierFactory.load_or_train_default()
        ctx["rows_processed"] = len(ref_df)
        ctx["details"] = {"model_name": clf.model_name}

    # 5. Simulação e Detecção de Drift + Registro MLflow
    detector = DriftDetector()
    mlflow_tracker = MLflowObservabilityTracker()

    for scenario in ("moderate_inflation", "severe_stagflation"):
        with pipe_logger.track_stage(f"5_drift_detection_{scenario}") as ctx:
            prod_df = load_or_create_production_dataset(
                scenario_name=scenario,
                classifier=clf,
            )
            summary, evidently_html, summary_json = detector.generate_evidently_reports(
                reference_df=ref_df,
                production_df=prod_df,
                classifier=clf,
                scenario_name=scenario,
            )
            exec_html = render_executive_drift_html(
                summary,
                settings.reports_dir / f"executive_drift_dashboard_{scenario}.html",
            )
            ctx["status"] = (
                "DRIFT_ALERT" if summary.dataset_drift_detected else "SUCCESS"
            )
            ctx["rows_processed"] = len(prod_df)
            ctx["drift_alerts"] = summary.drifted_features_count
            ctx["details"] = {
                "overall_severity": summary.overall_severity,
                "max_psi": summary.max_psi,
            }

            run_id = mlflow_tracker.log_monitoring_cycle(
                run_name=f"credit-guard-{clf.model_name}-{scenario}",
                classifier=clf,
                reference_df=ref_df,
                clean_contract=clean_res,
                corrupted_contract=corrupted_res,
                drift_summary=summary,
                privacy_report=privacy_report,
                artifact_paths=[
                    evidently_html,
                    exec_html,
                    summary_json,
                    spec_path,
                    val_report_path,
                    privacy_path,
                ],
            )
            logger.info(
                "Run MLflow registrado com sucesso (scenario=%s, run_id=%s)",
                scenario,
                run_id,
            )


if __name__ == "__main__":
    main()

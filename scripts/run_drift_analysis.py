"""Script CLI da Etapa 2: Simulação e Detecção Estatística de Drift com Evidently AI."""

from __future__ import annotations

import logging

from credit_guard.data.loader import load_or_create_reference_dataset
from credit_guard.drift.detector import DriftDetector, render_executive_drift_html
from credit_guard.drift.simulator import load_or_create_production_dataset
from credit_guard.model.factory import ClassifierFactory
from credit_guard.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Executa a simulação de produção e gera relatórios HTML do Evidently AI."""
    settings = get_settings()
    reference_df = load_or_create_reference_dataset()
    clf = ClassifierFactory.load_or_train_default()

    detector = DriftDetector()
    for scenario in ("moderate_inflation", "severe_stagflation"):
        logger.info("Simulando lote de produção para cenário '%s'...", scenario)
        prod_df = load_or_create_production_dataset(
            scenario_name=scenario,
            classifier=clf,
        )
        summary, evidently_html, summary_json = detector.generate_evidently_reports(
            reference_df=reference_df,
            production_df=prod_df,
            classifier=clf,
            scenario_name=scenario,
        )
        exec_html = render_executive_drift_html(
            summary,
            settings.reports_dir / f"executive_drift_dashboard_{scenario}.html",
        )
        logger.info(
            "Cenário '%s' | Severidade=%s | Drift=%d/%d (%s) | Max PSI=%.4f",
            scenario,
            summary.overall_severity,
            summary.drifted_features_count,
            summary.total_features_evaluated,
            ", ".join(summary.drifted_feature_names),
            summary.max_psi,
        )
        if summary.concept_drift is not None:
            cd = summary.concept_drift
            logger.info(
                "Concept Drift '%s' | ROC-AUC: %.4f -> %.4f | Default Real=%.2f%% vs "
                "Previsto=%.2f%% | CUSUM Alerta=%s (score=%.2f)",
                scenario,
                cd.reference_roc_auc,
                cd.production_roc_auc,
                cd.production_actual_default_rate * 100,
                cd.production_predicted_default_rate * 100,
                cd.cusum_alert_triggered,
                cd.cusum_max_score,
            )
        logger.info("Relatório Evidently AI salvo em: %s", evidently_html)
        logger.info("Painel executivo HTML salvo em: %s", exec_html)
        logger.info("Resumo estatístico JSON salvo em: %s", summary_json)


if __name__ == "__main__":
    main()

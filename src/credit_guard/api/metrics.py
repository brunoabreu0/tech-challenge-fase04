"""Métricas Prometheus para observabilidade de inferência, contratos e drift."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from credit_guard.drift.detector import DriftAnalysisSummary

HTTP_REQUESTS_TOTAL = Counter(
    "credit_api_http_requests_total",
    "Total de requisições HTTP processadas pela API de Credit Scoring",
    ["method", "endpoint", "http_status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "credit_api_http_request_duration_seconds",
    "Duração das requisições HTTP em segundos",
    ["method", "endpoint"],
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "credit_model_inference_duration_seconds",
    "Latência de inferência do modelo de risco de crédito em segundos",
    ["model_type"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

PREDICTIONS_TOTAL = Counter(
    "credit_model_predictions_total",
    "Total de decisões de crédito emitidas pelo modelo",
    ["decision", "risk_tier"],
)

CONTRACT_VALIDATIONS_TOTAL = Counter(
    "credit_data_contract_validations_total",
    "Total de lotes/registros avaliados pelo contrato Pandera",
    ["status"],
)

CONTRACT_VIOLATIONS_BY_COLUMN_TOTAL = Counter(
    "credit_data_contract_violations_by_column_total",
    "Total de violações de qualidade de dados interceptadas por coluna",
    ["column"],
)

DRIFT_FEATURE_PSI_GAUGE = Gauge(
    "credit_drift_feature_psi",
    "Population Stability Index (PSI) atual por variável e cenário",
    ["feature", "scenario"],
)

DRIFT_FEATURE_KS_STAT_GAUGE = Gauge(
    "credit_drift_feature_ks_statistic",
    "Estatística Kolmogorov-Smirnov (KS) por variável numérica e cenário",
    ["feature", "scenario"],
)

DRIFT_OVERALL_MAX_PSI_GAUGE = Gauge(
    "credit_drift_overall_max_psi",
    "Maior valor de PSI observado no lote de produção por cenário",
    ["scenario"],
)

DRIFT_DETECTED_FEATURES_GAUGE = Gauge(
    "credit_drift_detected_features_count",
    "Quantidade de features com drift estatístico detectado por cenário",
    ["scenario"],
)

CONCEPT_DRIFT_ROC_AUC_GAUGE = Gauge(
    "credit_concept_drift_roc_auc",
    "ROC-AUC comparativo entre Referência e Produção por cenário",
    ["dataset_split", "scenario"],
)

CONCEPT_DRIFT_RISK_GAP_GAUGE = Gauge(
    "credit_concept_drift_unpredicted_risk_gap",
    "Diferença entre inadimplência real e probabilidade média prevista em produção",
    ["scenario"],
)

CONCEPT_DRIFT_CUSUM_SCORE_GAUGE = Gauge(
    "credit_concept_drift_cusum_score",
    "Score acumulado do detector de mudança de regime CUSUM por cenário",
    ["scenario"],
)

LGPD_DISPARATE_IMPACT_RATIO_GAUGE = Gauge(
    "credit_lgpd_fairness_disparate_impact_ratio",
    "Razão de impacto adverso entre faixas etárias (meta >= 0.80 LGPD)",
)


def publish_drift_metrics_to_prometheus(summary: DriftAnalysisSummary) -> None:
    """Publica as métricas estatísticas de Data e Concept Drift no Prometheus."""
    scenario = summary.scenario_name
    DRIFT_OVERALL_MAX_PSI_GAUGE.labels(scenario=scenario).set(summary.max_psi)
    DRIFT_DETECTED_FEATURES_GAUGE.labels(scenario=scenario).set(
        summary.drifted_features_count
    )

    for fm in summary.feature_metrics:
        DRIFT_FEATURE_PSI_GAUGE.labels(feature=fm.feature, scenario=scenario).set(
            fm.psi
        )
        if fm.ks_statistic is not None:
            DRIFT_FEATURE_KS_STAT_GAUGE.labels(
                feature=fm.feature, scenario=scenario
            ).set(fm.ks_statistic)

    if summary.concept_drift is not None:
        cd = summary.concept_drift
        CONCEPT_DRIFT_ROC_AUC_GAUGE.labels(
            dataset_split="reference", scenario=scenario
        ).set(cd.reference_roc_auc)
        CONCEPT_DRIFT_ROC_AUC_GAUGE.labels(
            dataset_split="production", scenario=scenario
        ).set(cd.production_roc_auc)
        CONCEPT_DRIFT_RISK_GAP_GAUGE.labels(scenario=scenario).set(
            cd.unpredicted_risk_gap
        )
        CONCEPT_DRIFT_CUSUM_SCORE_GAUGE.labels(scenario=scenario).set(
            cd.cusum_max_score
        )

"""Aplicação FastAPI de Credit Scoring com Contratos Pandera, Drift e LGPD."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from credit_guard.api.metrics import (
    CONTRACT_VALIDATIONS_TOTAL,
    CONTRACT_VIOLATIONS_BY_COLUMN_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    INFERENCE_LATENCY_SECONDS,
    LGPD_DISPARATE_IMPACT_RATIO_GAUGE,
    PREDICTIONS_TOTAL,
    publish_drift_metrics_to_prometheus,
)
from credit_guard.api.schemas import (
    BatchValidationRequest,
    CreditApplicationInput,
    CreditPredictionResponse,
    DriftTriggerRequest,
    HealthResponse,
)
from credit_guard.data.contracts import (
    ContractValidationResult,
    DataContractValidator,
    generate_corrupted_batch,
)
from credit_guard.data.loader import (
    generate_credit_dataset,
    load_or_create_reference_dataset,
)
from credit_guard.data.privacy import PrivacyGuard
from credit_guard.drift.detector import (
    DriftAnalysisSummary,
    DriftDetector,
    render_executive_drift_html,
)
from credit_guard.drift.simulator import DriftSimulator
from credit_guard.model.base import BaseClassifier
from credit_guard.model.factory import ClassifierFactory
from credit_guard.observability.logger import PipelineObservabilityTracker
from credit_guard.settings import get_settings

logger = logging.getLogger(__name__)

_TRACKED_ENDPOINTS: set[str] = {
    "/health",
    "/predict",
    "/validate",
    "/drift/analyze",
    "/observability/pipeline",
    "/governance/lgpd",
    "/metrics",
}


class AppState:
    """Mantém estado em memória para classificador, referência e drift."""

    classifier: BaseClassifier | None = None
    reference_df: pd.DataFrame | None = None
    raw_reference_df: pd.DataFrame | None = None
    privacy_report: dict[str, Any] = {}
    drift_summaries: dict[str, DriftAnalysisSummary] = {}
    pipeline_tracker: PipelineObservabilityTracker | None = None


state = AppState()


def _ensure_initialized() -> None:
    """Inicializa artefatos em memória caso ainda não estejam carregados."""
    if state.classifier is not None and state.reference_df is not None:
        return

    settings = get_settings()
    tracker = PipelineObservabilityTracker()
    guard = PrivacyGuard()
    validator = DataContractValidator(require_target=False)

    with tracker.track_stage("startup_load_reference_and_lgpd") as ctx:
        raw_df = generate_credit_dataset(
            n_samples=settings.reference_samples,
            seed=settings.random_seed,
            include_pii=True,
        )
        ref_df = load_or_create_reference_dataset()
        priv_report = guard.generate_privacy_audit_report(raw_df, ref_df)
        fairness = priv_report.get("fairness_and_non_discrimination", {})
        if "disparate_impact_ratio" in fairness:
            LGPD_DISPARATE_IMPACT_RATIO_GAUGE.set(
                float(fairness["disparate_impact_ratio"])
            )
        ctx["rows_processed"] = len(ref_df)

    with tracker.track_stage("startup_reference_contract_check") as ctx:
        clean_res = validator.validate_batch(ref_df)
        CONTRACT_VALIDATIONS_TOTAL.labels(status="valid").inc(clean_res.total_rows)
        ctx["rows_processed"] = clean_res.total_rows

    with tracker.track_stage("startup_corrupted_gatekeeper_warmup") as ctx:
        corrupted_df = generate_corrupted_batch(ref_df, n_samples=60, seed=42)
        corrupted_res = validator.validate_batch(corrupted_df)
        CONTRACT_VALIDATIONS_TOTAL.labels(status="blocked").inc(
            corrupted_res.failed_rows_count
        )
        for col, cnt in corrupted_res.violations_by_column.items():
            CONTRACT_VIOLATIONS_BY_COLUMN_TOTAL.labels(column=col).inc(cnt)
        ctx["status"] = "BLOCKED"
        ctx["rows_processed"] = corrupted_res.total_rows
        ctx["contract_violations"] = corrupted_res.violation_count

    clf = ClassifierFactory.load_or_train_default()
    detector = DriftDetector()
    simulator = DriftSimulator(seed=settings.random_seed + 1000)

    for scenario in ("baseline_stationary", "moderate_inflation", "severe_stagflation"):
        with tracker.track_stage(f"startup_drift_{scenario}") as ctx:
            prod_df = simulator.simulate_production_dataset(
                n_samples=min(settings.production_samples, 1500),
                scenario_name=scenario,
                classifier=clf,
            )
            summary = detector.analyze(
                reference_df=ref_df,
                production_df=prod_df,
                classifier=clf,
                scenario_name=scenario,
            )
            publish_drift_metrics_to_prometheus(summary)
            state.drift_summaries[scenario] = summary
            ctx["rows_processed"] = len(prod_df)
            ctx["drift_alerts"] = summary.drifted_features_count
            ctx["status"] = (
                "DRIFT_ALERT" if summary.dataset_drift_detected else "SUCCESS"
            )

    state.raw_reference_df = raw_df
    state.reference_df = ref_df
    state.privacy_report = priv_report
    state.classifier = clf
    state.pipeline_tracker = tracker


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Ciclo de vida da aplicação FastAPI."""
    _ensure_initialized()
    yield


settings = get_settings()
app = FastAPI(
    title="CreditGuard — Observabilidade, Contratos de Dados, Drift & LGPD API",
    description=(
        "Tech Challenge Fase 4 (FIAP Pós-Tech 9MLET) — Sustentação, Contratos Pandera, "
        "Detecção de Data/Concept Drift com Evidently AI e Governança LGPD."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_http_middleware(request: Request, call_next: Any) -> Response:
    """Instrumenta latência e vazão HTTP filtrando rotas conhecidas contra ruído."""
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = time.perf_counter() - start

    path = request.url.path
    normalized_path = (
        "/drift/report"
        if path.startswith("/drift/report/")
        else ("/drift/dashboard" if path.startswith("/drift/dashboard/") else path)
    )
    if normalized_path in _TRACKED_ENDPOINTS or normalized_path in {
        "/drift/report",
        "/drift/dashboard",
    }:
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=normalized_path,
            http_status=str(response.status_code),
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=normalized_path,
        ).observe(elapsed)
    return response


@app.get("/health", response_model=HealthResponse, tags=["Monitoramento"])
def health_check() -> HealthResponse:
    """Verifica a saúde operacional do serviço, do modelo e dos detectores de drift."""
    _ensure_initialized()
    assert state.classifier is not None
    assert state.reference_df is not None
    assert state.pipeline_tracker is not None

    severe = state.drift_summaries.get("severe_stagflation")
    pipe_health = state.pipeline_tracker.get_health_summary()
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        model_loaded=state.classifier.model_name,
        reference_samples=len(state.reference_df),
        pipeline_status=pipe_health["pipeline_status"],
        latest_drift_severity=severe.overall_severity if severe else "UNKNOWN",
        lgpd_compliant=bool(state.privacy_report.get("is_compliant", True)),
        details={
            "total_blocked_batches": pipe_health["total_blocked_batches"],
            "total_contract_violations": pipe_health["total_contract_violations"],
            "severe_stagflation_max_psi": severe.max_psi if severe else 0.0,
        },
    )


@app.get("/metrics", tags=["Monitoramento"])
def metrics_endpoint() -> Response:
    """Expõe métricas no formato texto do Prometheus."""
    _ensure_initialized()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=CreditPredictionResponse, tags=["Credit Scoring"])
def predict_credit_risk(payload: CreditApplicationInput) -> CreditPredictionResponse:
    """Valida o contrato via Pandera, aplica LGPD PrivacyGuard e roda inferência."""
    _ensure_initialized()
    assert state.classifier is not None
    assert state.pipeline_tracker is not None

    start = time.perf_counter()
    raw_df = pd.DataFrame([payload.model_dump()])
    guard = PrivacyGuard()
    masked_cpf = guard.mask_cpf(payload.cpf or "")
    gen_zip = guard.generalize_zip_code(payload.zip_code or "")
    sanitized_df = guard.sanitize_dataframe(raw_df, drop_direct_pii=True)

    validator = DataContractValidator(require_target=False)
    contract_res = validator.validate_batch(sanitized_df)
    if not contract_res.is_valid:
        CONTRACT_VALIDATIONS_TOTAL.labels(status="blocked").inc()
        for v in contract_res.violations:
            CONTRACT_VIOLATIONS_BY_COLUMN_TOTAL.labels(column=v.column).inc()
        state.pipeline_tracker.record_event(
            stage="realtime_predict_contract_gatekeeper",
            status="BLOCKED",
            duration_ms=(time.perf_counter() - start) * 1000.0,
            rows_processed=1,
            contract_violations=contract_res.violation_count,
            details={"violations": [v.model_dump() for v in contract_res.violations]},
        )
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Ingestão bloqueada pelo Contrato de Dados (Pandera).",
                "contract_validation": contract_res.model_dump(),
            },
        )

    CONTRACT_VALIDATIONS_TOTAL.labels(status="valid").inc()
    inf_start = time.perf_counter()
    prob = float(state.classifier.predict_proba(sanitized_df)[0])
    pred = int(state.classifier.predict(sanitized_df)[0])
    inf_elapsed = time.perf_counter() - inf_start
    INFERENCE_LATENCY_SECONDS.labels(model_type=state.classifier.model_name).observe(
        inf_elapsed
    )

    if prob < 0.20:
        risk_tier = "LOW"
        decision = "APPROVED"
    elif prob < 0.40:
        risk_tier = "MODERATE"
        decision = "APPROVED_WITH_LIMIT"
    else:
        risk_tier = "HIGH"
        decision = "REJECTED"

    PREDICTIONS_TOTAL.labels(decision=decision, risk_tier=risk_tier).inc()
    total_ms = round((time.perf_counter() - start) * 1000.0, 3)

    return CreditPredictionResponse(
        application_id=payload.application_id,
        customer_token=str(sanitized_df["customer_token"].iloc[0]),
        masked_cpf=masked_cpf,
        generalized_zip_code=gen_zip,
        model_name=state.classifier.model_name,
        default_probability=round(prob, 4),
        default_risk=pred,
        decision=decision,
        risk_tier=risk_tier,
        contract_valid=True,
        latency_ms=total_ms,
        lgpd_notice=(
            "Decisão sob base legal de Proteção ao Crédito (LGPD Art. 7º, X). "
            "PII direta pseudonimizada na origem (Privacy by Default). O titular "
            "possui direito à explicação e revisão nos termos do Art. 20º da LGPD."
        ),
    )


@app.post(
    "/validate",
    response_model=ContractValidationResult,
    tags=["Qualidade de Dados & Contratos"],
)
def validate_batch_endpoint(
    request: BatchValidationRequest | None = None,
    simulate_corrupted: bool = Query(
        default=False,
        description="Gera lote corrompido para demonstrar bloqueio Pandera",
    ),
) -> ContractValidationResult:
    """Executa o contrato de qualidade Pandera em um lote ou simulação de falha."""
    _ensure_initialized()
    assert state.reference_df is not None
    assert state.pipeline_tracker is not None

    validator = DataContractValidator(require_target=False)
    with state.pipeline_tracker.track_stage("api_batch_contract_validation") as ctx:
        if simulate_corrupted or request is None or len(request.records) == 0:
            batch_df = generate_corrupted_batch(
                state.reference_df, n_samples=80, seed=int(time.time()) % 10000
            )
        else:
            raw_df = pd.DataFrame([r.model_dump() for r in request.records])
            batch_df = PrivacyGuard().sanitize_dataframe(raw_df, drop_direct_pii=True)

        result = validator.validate_batch(batch_df)
        ctx["rows_processed"] = result.total_rows
        ctx["contract_violations"] = result.violation_count
        ctx["status"] = "BLOCKED" if result.blocked_ingestion else "SUCCESS"

        if result.blocked_ingestion:
            CONTRACT_VALIDATIONS_TOTAL.labels(status="blocked").inc(
                result.failed_rows_count
            )
            for col, cnt in result.violations_by_column.items():
                CONTRACT_VIOLATIONS_BY_COLUMN_TOTAL.labels(column=col).inc(cnt)
        else:
            CONTRACT_VALIDATIONS_TOTAL.labels(status="valid").inc(result.total_rows)

    return result


@app.post(
    "/drift/analyze",
    response_model=DriftAnalysisSummary,
    tags=["Detecção de Drift"],
)
def analyze_drift_endpoint(request: DriftTriggerRequest) -> DriftAnalysisSummary:
    """Executa simulação e testes estatísticos (PSI, KS, Wasserstein, CUSUM)."""
    _ensure_initialized()
    assert state.reference_df is not None
    assert state.classifier is not None
    assert state.pipeline_tracker is not None

    simulator = DriftSimulator(seed=int(time.time()) % 10000)
    try:
        prod_df = simulator.simulate_production_dataset(
            n_samples=request.n_samples,
            scenario_name=request.scenario_name,
            classifier=state.classifier,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detector = DriftDetector()
    with state.pipeline_tracker.track_stage(
        f"api_drift_analysis_{request.scenario_name}",
        rows_processed=len(prod_df),
    ) as ctx:
        summary = detector.analyze(
            reference_df=state.reference_df,
            production_df=prod_df,
            classifier=state.classifier,
            scenario_name=request.scenario_name,
        )
        publish_drift_metrics_to_prometheus(summary)
        state.drift_summaries[request.scenario_name] = summary
        ctx["drift_alerts"] = summary.drifted_features_count
        ctx["status"] = "DRIFT_ALERT" if summary.dataset_drift_detected else "SUCCESS"

    return summary


@app.get(
    "/drift/report/{scenario_name}",
    response_class=HTMLResponse,
    tags=["Detecção de Drift"],
)
def get_evidently_html_report(
    scenario_name: str = "severe_stagflation",
) -> HTMLResponse:
    """Retorna ou gera o relatório HTML interativo do Evidently AI para o cenário."""
    _ensure_initialized()
    assert state.reference_df is not None
    assert state.classifier is not None

    html_path = settings.reports_dir / f"evidently_drift_{scenario_name}.html"
    if not html_path.exists():
        simulator = DriftSimulator(seed=settings.random_seed + 1000)
        try:
            prod_df = simulator.simulate_production_dataset(
                n_samples=1200,
                scenario_name=scenario_name,
                classifier=state.classifier,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        detector = DriftDetector()
        summary, html_path, _ = detector.generate_evidently_reports(
            reference_df=state.reference_df,
            production_df=prod_df,
            classifier=state.classifier,
            scenario_name=scenario_name,
        )
        publish_drift_metrics_to_prometheus(summary)
        state.drift_summaries[scenario_name] = summary

    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get(
    "/drift/dashboard/{scenario_name}",
    response_class=HTMLResponse,
    tags=["Detecção de Drift"],
)
def get_executive_drift_dashboard(
    scenario_name: str = "severe_stagflation",
) -> HTMLResponse:
    """Retorna o painel executivo HTML de PSI, Kolmogorov-Smirnov e Concept Drift."""
    _ensure_initialized()
    assert state.reference_df is not None
    assert state.classifier is not None

    summary = state.drift_summaries.get(scenario_name)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cenário '{scenario_name}' não encontrado.",
        )

    exec_path = settings.reports_dir / f"executive_drift_dashboard_{scenario_name}.html"
    render_executive_drift_html(summary, exec_path)
    return HTMLResponse(content=exec_path.read_text(encoding="utf-8"))


@app.get("/observability/pipeline", tags=["Monitoramento"])
def get_pipeline_observability() -> dict[str, Any]:
    """Retorna os logs estruturados centralizados e métricas de saúde do pipeline."""
    _ensure_initialized()
    assert state.pipeline_tracker is not None
    return state.pipeline_tracker.get_health_summary()


@app.get("/governance/lgpd", tags=["Governança & LGPD"])
def get_lgpd_governance_audit() -> dict[str, Any]:
    """Retorna o relatório de auditoria de Privacidade (LGPD) e Mitigação de Vieses."""
    _ensure_initialized()
    return state.privacy_report

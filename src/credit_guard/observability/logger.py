"""Centralizador de logs estruturados (JSONL) e métricas operacionais do pipeline."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from credit_guard.settings import get_settings

logger = logging.getLogger(__name__)


class PipelineStageEvent(BaseModel):
    """Evento estruturado de execução de uma etapa do pipeline de MLOps."""

    model_config = ConfigDict(frozen=True)

    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    stage: str = Field(description="Nome da etapa (ex: ingestion, contract_validation)")
    status: str = Field(description="SUCCESS, BLOCKED, DRIFT_ALERT ou ERROR")
    duration_ms: float = Field(description="Latência da etapa em milissegundos")
    rows_processed: int = Field(default=0, description="Volume de linhas processadas")
    contract_violations: int = Field(
        default=0, description="Violações de contrato interceptadas"
    )
    drift_alerts: int = Field(
        default=0, description="Quantidade de variáveis com drift detectado"
    )
    details: dict[str, Any] = Field(default_factory=dict)


class PipelineObservabilityTracker:
    """Centraliza eventos de execução, falhas de ingestão e alertas de drift."""

    def __init__(self, reports_dir: Path | None = None) -> None:
        """Inicializa o rastreador apontando para o diretório de relatórios."""
        settings = get_settings()
        self.reports_dir = reports_dir or settings.reports_dir
        self.events_file = self.reports_dir / "pipeline_execution_events.jsonl"
        self.health_file = self.reports_dir / "pipeline_health_summary.json"
        self.events: list[PipelineStageEvent] = []

    def record_event(
        self,
        stage: str,
        status: str,
        duration_ms: float,
        rows_processed: int = 0,
        contract_violations: int = 0,
        drift_alerts: int = 0,
        details: dict[str, Any] | None = None,
    ) -> PipelineStageEvent:
        """Registra um evento estruturado em memória, no log e em arquivo JSONL."""
        event = PipelineStageEvent(
            stage=stage,
            status=status,
            duration_ms=round(duration_ms, 3),
            rows_processed=rows_processed,
            contract_violations=contract_violations,
            drift_alerts=drift_alerts,
            details=details or {},
        )
        self.events.append(event)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        with self.events_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.model_dump(), ensure_ascii=False) + "\n")

        logger.info(
            "PIPELINE_EVENT | stage=%s | status=%s | duration_ms=%.2f | rows=%d | "
            "violations=%d | drift_alerts=%d",
            event.stage,
            event.status,
            event.duration_ms,
            event.rows_processed,
            event.contract_violations,
            event.drift_alerts,
        )
        self.save_health_summary()
        return event

    @contextmanager
    def track_stage(
        self,
        stage: str,
        rows_processed: int = 0,
    ) -> Generator[dict[str, Any], None, None]:
        """Context manager que cronometra a latência da etapa e captura falhas."""
        ctx: dict[str, Any] = {
            "status": "SUCCESS",
            "rows_processed": rows_processed,
            "contract_violations": 0,
            "drift_alerts": 0,
            "details": {},
        }
        start = time.perf_counter()
        try:
            yield ctx
        except Exception as exc:
            ctx["status"] = "ERROR"
            ctx["details"]["error"] = str(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record_event(
                stage=stage,
                status=str(ctx["status"]),
                duration_ms=elapsed_ms,
                rows_processed=int(ctx["rows_processed"]),
                contract_violations=int(ctx["contract_violations"]),
                drift_alerts=int(ctx["drift_alerts"]),
                details=dict(ctx["details"]),
            )

    def get_health_summary(self) -> dict[str, Any]:
        """Consolida métricas operacionais de saúde do pipeline de dados e modelo."""
        all_events = self._load_all_persisted_events()
        total_events = len(all_events)
        if total_events == 0:
            return {
                "pipeline_status": "IDLE",
                "total_stage_executions": 0,
                "total_rows_processed": 0,
                "total_blocked_batches": 0,
                "total_contract_violations": 0,
                "total_drift_alerts": 0,
                "mean_stage_latency_ms": 0.0,
                "recent_events": [],
            }

        total_rows = sum(e.get("rows_processed", 0) for e in all_events)
        blocked_batches = sum(1 for e in all_events if e.get("status") == "BLOCKED")
        violations = sum(e.get("contract_violations", 0) for e in all_events)
        drift_alerts = sum(e.get("drift_alerts", 0) for e in all_events)
        errors = sum(1 for e in all_events if e.get("status") == "ERROR")
        mean_lat = float(
            sum(e.get("duration_ms", 0.0) for e in all_events) / total_events
        )

        if errors > 0:
            overall = "DEGRADED_ERROR"
        elif drift_alerts > 0:
            overall = "DRIFT_WARNING"
        else:
            overall = "HEALTHY"

        return {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "pipeline_status": overall,
            "total_stage_executions": total_events,
            "total_rows_processed": total_rows,
            "total_blocked_batches": blocked_batches,
            "total_contract_violations": violations,
            "total_drift_alerts": drift_alerts,
            "error_count": errors,
            "mean_stage_latency_ms": round(mean_lat, 3),
            "recent_events": all_events[-10:],
        }

    def save_health_summary(self) -> Path:
        """Persiste o resumo consolidado de saúde operacional em JSON."""
        summary = self.get_health_summary()
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.health_file.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return self.health_file

    def _load_all_persisted_events(self) -> list[dict[str, Any]]:
        """Carrega o histórico de eventos do arquivo JSONL se existir."""
        if not self.events_file.exists():
            return [e.model_dump() for e in self.events]
        loaded: list[dict[str, Any]] = []
        for line in self.events_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                loaded.append(json.loads(line))
        return loaded

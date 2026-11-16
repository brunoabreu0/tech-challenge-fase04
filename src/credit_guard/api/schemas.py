"""Schemas Pydantic v2 para requisições e respostas da API de Credit Scoring."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CreditApplicationInput(BaseModel):
    """Payload de entrada de proposta de crédito (inclui PII opcional)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "application_id": "APP-2026-000101",
                "cpf": "123.456.789-01",
                "full_name": "Cliente Exemplo",
                "zip_code": "01310-100",
                "age": 34,
                "annual_income": 78000.0,
                "employment_length_years": 6.5,
                "debt_to_income_ratio": 0.28,
                "credit_utilization": 0.32,
                "loan_amount": 18000.0,
                "interest_rate": 14.5,
                "num_delinquencies_2yrs": 0,
                "unemployment_rate_region": 7.2,
                "credit_score": 710,
                "home_ownership": "MORTGAGE",
                "loan_purpose": "DEBT_CONSOLIDATION",
            }
        }
    )

    application_id: str = Field(default="APP-LIVE-0001")
    cpf: str | None = Field(default="000.000.000-00")
    full_name: str | None = Field(default=None)
    zip_code: str | None = Field(default="01000-000")
    age: int = Field(description="Idade do solicitante (contrato exige > 18)")
    annual_income: float | None = Field(description="Renda anual em R$")
    employment_length_years: float = Field(default=3.0)
    debt_to_income_ratio: float = Field(default=0.25)
    credit_utilization: float = Field(default=0.30)
    loan_amount: float = Field(default=15000.0)
    interest_rate: float = Field(default=14.0)
    num_delinquencies_2yrs: int = Field(default=0)
    unemployment_rate_region: float = Field(default=7.5)
    credit_score: int = Field(default=680)
    home_ownership: str = Field(default="RENT")
    loan_purpose: str = Field(default="DEBT_CONSOLIDATION")


class BatchValidationRequest(BaseModel):
    """Requisição para validar uma lista de registros contra o contrato Pandera."""

    model_config = ConfigDict(extra="ignore")

    records: list[CreditApplicationInput] = Field(default_factory=list)


class CreditPredictionResponse(BaseModel):
    """Resposta da inferência de risco de crédito com metadados de governança LGPD."""

    model_config = ConfigDict(frozen=True)

    application_id: str
    customer_token: str
    masked_cpf: str
    generalized_zip_code: str
    model_name: str
    default_probability: float
    default_risk: int
    decision: str
    risk_tier: str
    contract_valid: bool
    latency_ms: float
    lgpd_notice: str


class DriftTriggerRequest(BaseModel):
    """Parâmetros para disparar análise estatística de Drift sob demanda."""

    model_config = ConfigDict(extra="ignore")

    scenario_name: str = Field(
        default="severe_stagflation",
        description="Cenário de simulação macroeconômica de Drift",
    )
    n_samples: int = Field(default=1500, ge=200, le=10000)


class HealthResponse(BaseModel):
    """Status operacional da API, do modelo e do monitoramento de Drift."""

    model_config = ConfigDict(frozen=True)

    status: str
    app_name: str
    version: str
    model_loaded: str
    reference_samples: int
    pipeline_status: str
    latest_drift_severity: str
    lgpd_compliant: bool
    details: dict[str, Any] = Field(default_factory=dict)

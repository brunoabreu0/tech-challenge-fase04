"""Testes de integração da API FastAPI (inferência, contratos, drift e LGPD)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from credit_guard.api.main import app


def test_api_health_and_metrics_endpoints() -> None:
    """Testa /, /health e /metrics garantindo exposição de Gauges de PSI e Contratos."""
    with TestClient(app) as client:
        portal_resp = client.get("/evidently")
        assert portal_resp.status_code == 200
        assert "Evidently AI & Drift Observatory" in portal_resp.text

        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        body = health_resp.json()
        assert body["status"] == "healthy"
        assert body["reference_samples"] >= 5000
        assert body["lgpd_compliant"] is True
        assert body["latest_drift_severity"] == "CRITICAL"

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        text = metrics_resp.text
        assert "credit_drift_feature_psi" in text
        assert "credit_data_contract_validations_total" in text
        assert "credit_concept_drift_roc_auc" in text


def test_api_predict_valid_and_contract_violation_blocked() -> None:
    """Testa predição válida com anonimização LGPD e bloqueio HTTP 422."""
    with TestClient(app) as client:
        valid_payload = {
            "application_id": "APP-TEST-001",
            "cpf": "123.456.789-01",
            "full_name": "Maria Teste",
            "zip_code": "01310-100",
            "age": 36,
            "annual_income": 92000.0,
            "employment_length_years": 8.0,
            "debt_to_income_ratio": 0.22,
            "credit_utilization": 0.25,
            "loan_amount": 15000.0,
            "interest_rate": 12.5,
            "num_delinquencies_2yrs": 0,
            "unemployment_rate_region": 6.8,
            "credit_score": 740,
            "home_ownership": "OWN",
            "loan_purpose": "HOME_IMPROVEMENT",
        }
        resp = client.post("/predict", json=valid_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["contract_valid"] is True
        assert data["masked_cpf"] == "***.456.789-**"
        assert data["generalized_zip_code"] == "013XX-XXX"
        assert 0.0 <= data["default_probability"] <= 1.0
        assert data["decision"] in {"APPROVED", "APPROVED_WITH_LIMIT", "REJECTED"}

        # Proposta violando o contrato Pandera (idade <= 18 e renda negativa)
        invalid_payload = dict(valid_payload)
        invalid_payload["application_id"] = "APP-BAD-002"
        invalid_payload["age"] = 16
        invalid_payload["annual_income"] = -500.0
        bad_resp = client.post("/predict", json=invalid_payload)
        assert bad_resp.status_code == 422
        assert (
            bad_resp.json()["detail"]["contract_validation"]["blocked_ingestion"]
            is True
        )


def test_api_validate_batch_and_drift_and_governance_endpoints() -> None:
    """Testa /validate, /drift/analyze, /drift/dashboard e /governance."""
    with TestClient(app) as client:
        # 1. Simulação de lote corrompido
        corrupted_resp = client.post("/validate?simulate_corrupted=true")
        assert corrupted_resp.status_code == 200
        assert corrupted_resp.json()["blocked_ingestion"] is True

        # 2. Lote válido customizado
        clean_batch = {
            "records": [
                {
                    "application_id": "APP-BATCH-1",
                    "age": 42,
                    "annual_income": 85000.0,
                    "employment_length_years": 10.0,
                    "debt_to_income_ratio": 0.20,
                    "credit_utilization": 0.18,
                    "loan_amount": 12000.0,
                    "interest_rate": 11.5,
                    "num_delinquencies_2yrs": 0,
                    "unemployment_rate_region": 7.0,
                    "credit_score": 755,
                    "home_ownership": "OWN",
                    "loan_purpose": "CREDIT_CARD",
                }
            ]
        }
        clean_resp = client.post("/validate", json=clean_batch)
        assert clean_resp.status_code == 200
        assert clean_resp.json()["is_valid"] is True

        # 3. Análise de Drift sob demanda
        drift_resp = client.post(
            "/drift/analyze",
            json={"scenario_name": "severe_stagflation", "n_samples": 400},
        )
        assert drift_resp.status_code == 200
        assert drift_resp.json()["dataset_drift_detected"] is True

        bad_drift = client.post(
            "/drift/analyze",
            json={"scenario_name": "cenario_invalido", "n_samples": 400},
        )
        assert bad_drift.status_code == 400

        # 4. Painel HTML executivo e Relatório Evidently HTML
        dash_resp = client.get("/drift/dashboard/severe_stagflation")
        assert dash_resp.status_code == 200
        assert "Relatório Estatístico de Data Drift" in dash_resp.text

        evidently_resp = client.get("/drift/report/severe_stagflation")
        assert evidently_resp.status_code == 200
        assert "<html" in evidently_resp.text.lower()

        # 5. Observabilidade do pipeline e Governança LGPD
        obs_resp = client.get("/observability/pipeline")
        assert obs_resp.status_code == 200
        assert obs_resp.json()["total_stage_executions"] >= 4

        lgpd_resp = client.get("/governance/lgpd")
        assert lgpd_resp.status_code == 200
        assert lgpd_resp.json()["is_compliant"] is True

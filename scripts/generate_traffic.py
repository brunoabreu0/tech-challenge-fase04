"""Gerador contínuo de tráfego realista para alimentar Prometheus e Grafana."""

from __future__ import annotations

import json
import logging
import os
import random
import time
import urllib.error
import urllib.request

from credit_guard.data.preprocessor import VALID_HOME_OWNERSHIP, VALID_LOAN_PURPOSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _post_json(url: str, payload: dict[str, object]) -> int:
    """Envia requisição POST JSON e retorna o código HTTP."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except Exception as exc:
        logger.warning("Falha de conexão com %s: %s", url, exc)
        return 0


def build_valid_application(seq: int) -> dict[str, object]:
    """Gera proposta de crédito válida variando perfis de risco (baixo, médio, alto)."""
    age = random.randint(21, 72)
    income = round(random.uniform(24000.0, 220000.0), 2)
    exp = round(min(random.uniform(0.5, 20.0), float(age - 18)), 1)
    dti = round(random.uniform(0.08, 0.78), 4)
    util = round(random.uniform(0.05, 0.90), 4)
    score = random.randint(380, 820)
    cpf_str = (
        f"{random.randint(100, 999)}.{random.randint(100, 999)}."
        f"{random.randint(100, 999)}-01"
    )
    return {
        "application_id": f"LIVE-OK-{int(time.time())}-{seq}",
        "cpf": cpf_str,
        "full_name": "Solicitante Simulado",
        "zip_code": f"{random.randint(10000, 99999)}-000",
        "age": age,
        "annual_income": income,
        "employment_length_years": exp,
        "debt_to_income_ratio": dti,
        "credit_utilization": util,
        "loan_amount": round(random.uniform(3000.0, 85000.0), 2),
        "interest_rate": round(random.uniform(9.5, 38.0), 2),
        "num_delinquencies_2yrs": random.choice([0, 0, 0, 1, 2, 4]),
        "unemployment_rate_region": round(random.uniform(5.0, 16.5), 2),
        "credit_score": score,
        "home_ownership": random.choice(VALID_HOME_OWNERSHIP),
        "loan_purpose": random.choice(VALID_LOAN_PURPOSES),
    }


def build_invalid_application(seq: int) -> dict[str, object]:
    """Gera proposta com anomalias propositais para exercitar o bloqueio Pandera."""
    payload = build_valid_application(seq)
    payload["application_id"] = f"LIVE-BAD-{int(time.time())}-{seq}"
    anomaly_type = seq % 4
    if anomaly_type == 0:
        payload["age"] = random.choice([15, 17, -2])
    elif anomaly_type == 1:
        payload["annual_income"] = -8000.0
    elif anomaly_type == 2:
        payload["debt_to_income_ratio"] = 1.45
    else:
        payload["credit_score"] = 960
    return payload


def main() -> None:
    """Loop contínuo de simulação de tráfego (predições, bloqueios e drift)."""
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
    interval = float(os.getenv("TRAFFIC_INTERVAL_SECONDS", "3.0"))
    logger.info(
        "Iniciando gerador de tráfego contra %s (intervalo=%.1fs)", base_url, interval
    )

    step = 0
    while True:
        step += 1
        # Envia 3 propostas válidas
        for i in range(3):
            status = _post_json(
                f"{base_url}/predict",
                build_valid_application(step * 10 + i),
            )
            logger.info("POST /predict (válido) -> HTTP %d", status)

        # A cada 3 ciclos, envia 1 proposta inválida (bloqueio Pandera -> 422)
        if step % 3 == 0:
            bad_status = _post_json(
                f"{base_url}/predict",
                build_invalid_application(step),
            )
            logger.info(
                "POST /predict (anomalia bloqueada pelo Pandera) -> HTTP %d",
                bad_status,
            )

        # A cada 10 ciclos, recalcula análise de drift para atualizar Gauges
        if step % 10 == 0:
            scenario = random.choice(["severe_stagflation", "moderate_inflation"])
            drift_status = _post_json(
                f"{base_url}/drift/analyze",
                {"scenario_name": scenario, "n_samples": 800},
            )
            logger.info("POST /drift/analyze (%s) -> HTTP %d", scenario, drift_status)

        time.sleep(interval)


if __name__ == "__main__":
    main()

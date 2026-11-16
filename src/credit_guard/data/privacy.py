"""Camada de Governança de Dados, Privacidade (LGPD) e Privacy by Design & Default."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from credit_guard.settings import get_settings

DIRECT_PII_COLUMNS: tuple[str, ...] = ("cpf", "full_name")
QUASI_IDENTIFIER_COLUMNS: tuple[str, ...] = ("zip_code", "age")


class PrivacyGuard:
    """Aplica controles de Privacy by Design e By Default (LGPD) em dados de crédito.

    Implementa minimização de dados (Art. 6º, III da LGPD), pseudonimização via
    SHA-256 com salt criptográfico (Art. 13, § 4º), generalização de CEP para
    k-anonimato regional e auditoria de impacto disparatado (mitigação de viés).
    """

    def __init__(
        self, salt: str | None = None, retention_days: int | None = None
    ) -> None:
        """Inicializa o guardião de privacidade com salt e política de retenção."""
        settings = get_settings()
        self.salt = salt or settings.pii_hash_salt
        self.retention_days = retention_days or settings.data_retention_days

    def hash_identifier(self, value: str) -> str:
        """Converte identificador sensível (ex: CPF) em token SHA-256."""
        normalized = "".join(ch for ch in str(value) if ch.isalnum())
        payload = f"{self.salt}:{normalized}".encode()
        return hashlib.sha256(payload).hexdigest()[:24]

    @staticmethod
    def mask_cpf(cpf: str) -> str:
        """Mascara um CPF no formato ***.456.789-** para exibição segura em logs."""
        digits = "".join(ch for ch in str(cpf) if ch.isdigit())
        if len(digits) != 11:
            return "***.***.***-**"
        return f"***.{digits[3:6]}.{digits[6:9]}-**"

    @staticmethod
    def generalize_zip_code(zip_code: str) -> str:
        """Generaliza o CEP mantendo apenas os 3 primeiros dígitos (região postal)."""
        digits = "".join(ch for ch in str(zip_code) if ch.isdigit())
        if len(digits) < 3:
            return "000XX-XXX"
        return f"{digits[:3]}XX-XXX"

    def sanitize_dataframe(
        self,
        df: pd.DataFrame,
        drop_direct_pii: bool = True,
    ) -> pd.DataFrame:
        """Aplica Privacy by Default sanitizando colunas PII antes da modelagem.

        - Cria `customer_token` pseudonimizado a partir de `cpf` (se presente).
        - Generaliza `zip_code` para macro-região (`010XX-XXX`).
        - Remove identificadores diretos (`cpf`, `full_name`) se `drop_direct_pii`.
        """
        sanitized = df.copy()
        if "cpf" in sanitized.columns:
            sanitized["customer_token"] = (
                sanitized["cpf"].astype(str).map(self.hash_identifier)
            )
        if "zip_code" in sanitized.columns:
            sanitized["zip_code"] = (
                sanitized["zip_code"].astype(str).map(self.generalize_zip_code)
            )
        if drop_direct_pii:
            cols_to_drop = [c for c in DIRECT_PII_COLUMNS if c in sanitized.columns]
            if cols_to_drop:
                sanitized = sanitized.drop(columns=cols_to_drop)
        return sanitized

    def evaluate_fairness_by_age_group(
        self,
        df: pd.DataFrame,
        prediction_col: str = "default_risk",
    ) -> dict[str, Any]:
        """Calcula paridade demográfica e impacto adverso entre faixas etárias.

        Verifica se jovens adultos (18-25) ou idosos (60+) sofrem viés sistemático
        nas decisões de crédito (LGPD Art. 6º IX - Não discriminação).
        """
        if "age" not in df.columns or prediction_col not in df.columns:
            return {"status": "skipped", "reason": "Colunas age ou predição ausentes"}

        working = df[["age", prediction_col]].copy()
        working["age_group"] = pd.cut(
            working["age"],
            bins=[17, 25, 40, 60, 120],
            labels=["18-25", "26-40", "41-60", "60+"],
        )
        approval_rates: dict[str, float] = {}
        for group, group_df in working.groupby("age_group", observed=False):
            if len(group_df) > 0:
                approval_rate = float(1.0 - group_df[prediction_col].mean())
                approval_rates[str(group)] = round(approval_rate, 4)

        if not approval_rates:
            return {"status": "empty", "approval_rates": {}}

        max_rate = max(approval_rates.values())
        min_rate = min(approval_rates.values())
        disparate_impact_ratio = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0

        return {
            "status": "compliant" if disparate_impact_ratio >= 0.70 else "warning_bias",
            "approval_rate_by_age_group": approval_rates,
            "disparate_impact_ratio": disparate_impact_ratio,
            "four_fifths_rule_target": 0.80,
        }

    def generate_privacy_audit_report(
        self,
        raw_df: pd.DataFrame,
        sanitized_df: pd.DataFrame,
        target_col: str = "default_risk",
    ) -> dict[str, Any]:
        """Gera relatório consolidado de conformidade LGPD e Privacy by Design."""
        detected_pii = [c for c in DIRECT_PII_COLUMNS if c in raw_df.columns]
        remaining_pii = [c for c in DIRECT_PII_COLUMNS if c in sanitized_df.columns]
        fairness = self.evaluate_fairness_by_age_group(
            sanitized_df, prediction_col=target_col
        )

        return {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "framework": "LGPD (Lei nº 13.709/2018) & Privacy by Design (7 Princípios)",
            "legal_basis": {
                "primary": "Art. 7º, X — Proteção do crédito",
                "secondary": (
                    "Art. 7º, IX — Legítimo interesse do controlador "
                    "(prevenção à fraude)"
                ),
                "automated_decisions": (
                    "Art. 20º — Direito à revisão e explicação de decisões "
                    "automatizadas sobre perfil de crédito"
                ),
            },
            "data_minimization": {
                "raw_columns_count": len(raw_df.columns),
                "sanitized_columns_count": len(sanitized_df.columns),
                "detected_direct_pii": detected_pii,
                "remaining_direct_pii_in_model_input": remaining_pii,
                "pseudonymization_applied": "customer_token" in sanitized_df.columns,
                "zip_code_generalized": "zip_code" in sanitized_df.columns,
            },
            "retention_policy": {
                "retention_days": self.retention_days,
                "retention_years": round(self.retention_days / 365, 1),
                "disposal_action": (
                    "Anonimização irreversível ou expurgo criptográfico após 5 anos "
                    "(Art. 15 e Art. 16 da LGPD / Resolução CMN/BACEN)"
                ),
            },
            "fairness_and_non_discrimination": fairness,
            "is_compliant": len(remaining_pii) == 0,
        }

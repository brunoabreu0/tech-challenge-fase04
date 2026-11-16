"""Contratos de Qualidade de Dados com Pandera e Data Contract Specification (YAML)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
import yaml
from pandera.errors import SchemaErrors
from pydantic import BaseModel, ConfigDict, Field

from credit_guard.data.preprocessor import (
    VALID_HOME_OWNERSHIP,
    VALID_LOAN_PURPOSES,
)


class ContractViolationItem(BaseModel):
    """Representa uma violação individual detectada pelo contrato Pandera."""

    model_config = ConfigDict(frozen=True)

    column: str = Field(description="Coluna ou escopo onde ocorreu a falha")
    check: str = Field(description="Regra do contrato violada")
    failure_case: str = Field(description="Valor anômalo interceptado")
    row_index: int | None = Field(default=None, description="Índice da linha no lote")


class ContractValidationResult(BaseModel):
    """Resultado consolidado da validação de um lote contra o Contrato de Dados."""

    model_config = ConfigDict(frozen=True)

    contract_name: str = Field(default="credit_scoring_input_contract_v1")
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    is_valid: bool = Field(description="Indica se o lote cumpriu 100% do contrato")
    blocked_ingestion: bool = Field(
        description="True quando o lote foi bloqueado pelo gatekeeper de qualidade",
    )
    total_rows: int = Field(description="Total de registros avaliados no lote")
    failed_rows_count: int = Field(description="Quantidade de linhas com violação")
    violation_count: int = Field(description="Total de regras violadas no lote")
    violations_by_column: dict[str, int] = Field(default_factory=dict)
    violations: list[ContractViolationItem] = Field(default_factory=list)


def build_credit_dataframe_schema(require_target: bool = False) -> pa.DataFrameSchema:
    """Constrói o schema formal `DataFrameSchema` do Pandera (Schema-as-Code).

    Regras rígidas implementadas (superando o mínimo obrigatório de 3 regras):
    1. Sem nulos (`nullable=False`) em todas as variáveis críticas de decisão.
    2. Sem duplicatas em `application_id` (`unique=True`) e nas linhas completas.
    3. Range estrito de idade: `age > 18` e `age <= 100`.
    4. Limites estritos de renda anual: `annual_income >= 1000.0` e `<= 10_000_000.0`.
    5. Tempo de emprego não negativo (`0 <= employment_length_years <= 65`).
    6. Razão dívida/renda (`debt_to_income_ratio`) restrita a `[0.0, 1.0]`.
    7. Utilização de crédito (`credit_utilization`) restrita a `[0.0, 1.0]`.
    8. Valor do empréstimo (`loan_amount`) entre `R$ 500` e `R$ 500.000`.
    9. Score de crédito (`credit_score`) restrito à faixa FICO/Serasa `[300, 850]`.
    10. Domínio categórico fechado para `home_ownership` e `loan_purpose`.
    11. Regra cruzada: `employment_length_years <= age - 18`.
    """
    columns: dict[str, pa.Column] = {
        "application_id": pa.Column(
            str,
            nullable=False,
            unique=True,
            coerce=True,
            description="Identificador único da proposta de crédito (sem duplicatas)",
        ),
        "age": pa.Column(
            int,
            checks=[
                pa.Check.gt(18, error="age_must_be_greater_than_18"),
                pa.Check.le(100, error="age_must_be_at_most_100"),
            ],
            nullable=False,
            coerce=True,
            description="Idade do solicitante em anos completos (> 18 e <= 100)",
        ),
        "annual_income": pa.Column(
            float,
            checks=[
                pa.Check.ge(1000.0, error="annual_income_below_minimum_1000"),
                pa.Check.le(10_000_000.0, error="annual_income_exceeds_maximum_10M"),
            ],
            nullable=False,
            coerce=True,
            description="Renda anual declarada/comprovada em R$ (não nula)",
        ),
        "employment_length_years": pa.Column(
            float,
            checks=[
                pa.Check.ge(0.0, error="employment_length_negative"),
                pa.Check.le(65.0, error="employment_length_exceeds_65"),
            ],
            nullable=False,
            coerce=True,
            description="Tempo de vínculo empregatício em anos",
        ),
        "debt_to_income_ratio": pa.Column(
            float,
            checks=[
                pa.Check.ge(0.0, error="dti_below_0"),
                pa.Check.le(1.0, error="dti_above_1"),
            ],
            nullable=False,
            coerce=True,
            description="Comprometimento de renda (0.0 a 1.0)",
        ),
        "credit_utilization": pa.Column(
            float,
            checks=[
                pa.Check.ge(0.0, error="credit_utilization_below_0"),
                pa.Check.le(1.0, error="credit_utilization_above_1"),
            ],
            nullable=False,
            coerce=True,
            description="Taxa de utilização de limite rotativo (0.0 a 1.0)",
        ),
        "loan_amount": pa.Column(
            float,
            checks=[
                pa.Check.ge(500.0, error="loan_amount_below_500"),
                pa.Check.le(500_000.0, error="loan_amount_above_500k"),
            ],
            nullable=False,
            coerce=True,
            description="Valor solicitado de empréstimo em R$",
        ),
        "interest_rate": pa.Column(
            float,
            checks=[
                pa.Check.ge(1.0, error="interest_rate_below_1pct"),
                pa.Check.le(100.0, error="interest_rate_above_100pct"),
            ],
            nullable=False,
            coerce=True,
            description="Taxa de juros anualizada (%)",
        ),
        "num_delinquencies_2yrs": pa.Column(
            int,
            checks=[
                pa.Check.ge(0, error="delinquencies_negative"),
                pa.Check.le(50, error="delinquencies_above_50"),
            ],
            nullable=False,
            coerce=True,
            description="Ocorrências de atraso nos últimos 24 meses",
        ),
        "unemployment_rate_region": pa.Column(
            float,
            checks=[
                pa.Check.ge(0.0, error="unemployment_rate_negative"),
                pa.Check.le(50.0, error="unemployment_rate_above_50pct"),
            ],
            nullable=False,
            coerce=True,
            description="Taxa de desemprego regional (%)",
        ),
        "credit_score": pa.Column(
            int,
            checks=[
                pa.Check.ge(300, error="credit_score_below_300"),
                pa.Check.le(850, error="credit_score_above_850"),
            ],
            nullable=False,
            coerce=True,
            description="Score de bureau de crédito (300 a 850)",
        ),
        "home_ownership": pa.Column(
            str,
            checks=pa.Check.isin(
                list(VALID_HOME_OWNERSHIP),
                error="invalid_home_ownership_category",
            ),
            nullable=False,
            description="Situação de moradia",
        ),
        "loan_purpose": pa.Column(
            str,
            checks=pa.Check.isin(
                list(VALID_LOAN_PURPOSES),
                error="invalid_loan_purpose_category",
            ),
            nullable=False,
            description="Finalidade do empréstimo",
        ),
    }

    if require_target:
        columns["default_risk"] = pa.Column(
            int,
            checks=pa.Check.isin([0, 1], error="invalid_target_label"),
            nullable=False,
            coerce=True,
            description="Rótulo binário de inadimplência (0 ou 1)",
        )

    return pa.DataFrameSchema(
        columns=columns,
        checks=[
            pa.Check(
                lambda df: df["employment_length_years"] <= (df["age"] - 18),
                name="employment_consistent_with_adult_age",
                error="employment_length_exceeds_adult_working_age",
            ),
        ],
        strict=False,
        unique=["application_id"],
        name="credit_scoring_input_contract_v1",
    )


class DataContractValidator:
    """Gatekeeper de qualidade que valida lotes Pandas usando Pandera (`lazy=True`)."""

    def __init__(self, require_target: bool = False) -> None:
        """Inicializa o validador com o schema Pandera configurado."""
        self.require_target = require_target
        self.schema = build_credit_dataframe_schema(require_target=require_target)

    def validate_batch(self, df: pd.DataFrame) -> ContractValidationResult:
        """Executa validação lazy=True capturando todas as anomalias do lote."""
        try:
            self.schema.validate(df, lazy=True)
            return ContractValidationResult(
                is_valid=True,
                blocked_ingestion=False,
                total_rows=len(df),
                failed_rows_count=0,
                violation_count=0,
                violations_by_column={},
                violations=[],
            )
        except SchemaErrors as exc:
            failure_df = exc.failure_cases
            violations: list[ContractViolationItem] = []
            by_col: dict[str, int] = {}
            failed_indices: set[int] = set()

            for _, row in failure_df.iterrows():
                col = str(row.get("column") or "dataframe")
                check_name = str(row.get("check") or "schema_check")
                failure_val = str(row.get("failure_case"))
                idx_raw = row.get("index")
                row_idx: int | None = None
                if pd.notna(idx_raw) and isinstance(idx_raw, (int, np.integer)):
                    row_idx = int(idx_raw)
                    failed_indices.add(row_idx)

                by_col[col] = by_col.get(col, 0) + 1
                violations.append(
                    ContractViolationItem(
                        column=col,
                        check=check_name,
                        failure_case=failure_val,
                        row_index=row_idx,
                    )
                )

            return ContractValidationResult(
                is_valid=False,
                blocked_ingestion=True,
                total_rows=len(df),
                failed_rows_count=len(failed_indices) or min(len(violations), len(df)),
                violation_count=len(violations),
                violations_by_column=by_col,
                violations=violations[:100],
            )


def generate_corrupted_batch(
    reference_df: pd.DataFrame,
    n_samples: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Cria um lote de dados propositalmente corrompido para testar o bloqueio.

    Injeta 7 tipos de falhas críticas de qualidade de dados:
    1. Idade menor ou igual a 18 anos (`age = 16`, `age = -5`).
    2. Renda anual nula (`NaN`) e negativa (`-15000.0`).
    3. Duplicatas de `application_id`.
    4. Razão dívida/renda acima do limite máximo (`debt_to_income_ratio = 1.65`).
    5. Score de crédito fora da faixa `[300, 850]` (`credit_score = 990` e `120`).
    6. Categoria desconhecida em `home_ownership` (`"SQUATTER"`).
    7. Inconsistência cruzada: `employment_length_years = 20` para `age = 22`.
    """
    rng = np.random.default_rng(seed)
    corrupted = (
        reference_df.sample(n=min(n_samples, len(reference_df)), random_state=seed)
        .copy()
        .reset_index(drop=True)
    )

    # 1. Idade inválida (<= 18 e negativa)
    corrupted.loc[0, "age"] = 16
    corrupted.loc[1, "age"] = -5

    # 2. Renda anual nula e abaixo do mínimo
    corrupted.loc[2, "annual_income"] = np.nan
    corrupted.loc[3, "annual_income"] = -15000.0

    # 3. Duplicata de application_id
    corrupted.loc[4, "application_id"] = corrupted.loc[0, "application_id"]

    # 4. DTI fora do intervalo [0, 1]
    corrupted.loc[5, "debt_to_income_ratio"] = 1.65

    # 5. Credit score fora de [300, 850]
    corrupted.loc[6, "credit_score"] = 990
    corrupted.loc[7, "credit_score"] = 120

    # 6. Categoria inválida
    corrupted.loc[8, "home_ownership"] = "INVALID_STATUS"

    # 7. Tempo de emprego incompatível com a idade adulta
    corrupted.loc[9, "age"] = 22
    corrupted.loc[9, "employment_length_years"] = 15.0

    # Embaralha alguns índices adicionais com nulos aleatórios
    extra_null_rows = rng.choice(
        range(10, min(30, len(corrupted))), size=3, replace=False
    )
    for idx in extra_null_rows:
        corrupted.loc[int(idx), "annual_income"] = np.nan

    return corrupted


def export_data_contract_specification(output_path: Path) -> Path:
    """Exporta a especificação formal do Contrato de Dados em padrão YAML (DCS)."""
    spec: dict[str, Any] = {
        "dataContractSpecification": "1.1.0",
        "id": "urn:datacontract:fintech:credit-scoring-input:v1",
        "info": {
            "title": "Credit Scoring Input Data Contract",
            "version": "1.0.0",
            "description": (
                "Contrato formal de qualidade de dados para o pipeline de concessão "
                "de crédito (Pandera + Pydantic v2 + LGPD Privacy by Design)."
            ),
            "owner": "ML Engineering Team — Grupo 118 (FIAP 9MLET)",
        },
        "terms": {
            "usage": "Inferência e retreinamento do modelo de risco de crédito",
            "retention": "1825 dias (5 anos - Resolução CMN/BACEN e LGPD Art. 16)",
            "privacy": "PII direta (CPF, nome) removida/pseudonimizada na origem",
        },
        "quality": {
            "engine": "pandera",
            "validationMode": "lazy",
            "onFailure": "block_ingestion_and_alert",
            "rules": [
                {"name": "no_nulls_in_critical_features", "severity": "critical"},
                {"name": "unique_application_id", "severity": "critical"},
                {"name": "age_strictly_greater_than_18", "severity": "critical"},
                {"name": "annual_income_between_1k_and_10m", "severity": "critical"},
                {"name": "dti_and_utilization_between_0_and_1", "severity": "critical"},
                {"name": "credit_score_between_300_and_850", "severity": "critical"},
                {"name": "employment_consistent_with_age", "severity": "high"},
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def save_validation_report_json(
    clean_result: ContractValidationResult,
    corrupted_result: ContractValidationResult,
    output_path: Path,
) -> Path:
    """Salva o relatório JSON de execução dos contratos (lote limpo vs corrompido)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_batch_validation": clean_result.model_dump(),
        "corrupted_batch_validation": corrupted_result.model_dump(),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path

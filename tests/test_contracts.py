"""Testes unitários para a Etapa 1: Contratos de Dados com Pandera."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_guard.data.contracts import (
    DataContractValidator,
    export_data_contract_specification,
    generate_corrupted_batch,
    save_validation_report_json,
)


def test_clean_reference_batch_passes_contract(
    reference_credit_df: pd.DataFrame,
) -> None:
    """Garante que o dataset de referência limpo passe 100% no contrato Pandera."""
    validator = DataContractValidator(require_target=True)
    result = validator.validate_batch(reference_credit_df)
    assert result.is_valid is True
    assert result.blocked_ingestion is False
    assert result.violation_count == 0
    assert result.total_rows == len(reference_credit_df)


def test_corrupted_batch_is_intercepted_and_blocked(
    reference_credit_df: pd.DataFrame,
) -> None:
    """Verifica que lote com anomalias (idade <= 18, nulos, duplicatas) é bloqueado."""
    validator = DataContractValidator(require_target=True)
    corrupted_df = generate_corrupted_batch(reference_credit_df, n_samples=80, seed=42)
    result = validator.validate_batch(corrupted_df)

    assert result.is_valid is False
    assert result.blocked_ingestion is True
    assert result.violation_count >= 10
    assert result.failed_rows_count >= 8

    # Verifica que as regras críticas obrigatórias foram acionadas
    violated_cols = set(result.violations_by_column.keys())
    assert "age" in violated_cols
    assert "annual_income" in violated_cols
    assert "application_id" in violated_cols
    assert "debt_to_income_ratio" in violated_cols
    assert "credit_score" in violated_cols


def test_export_contract_spec_and_validation_json(
    reference_credit_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Testa a exportação do YAML Data Contract Specification e relatório JSON."""
    validator = DataContractValidator(require_target=True)
    clean_res = validator.validate_batch(reference_credit_df)
    corrupted_res = validator.validate_batch(
        generate_corrupted_batch(reference_credit_df, n_samples=40)
    )

    yaml_path = export_data_contract_specification(tmp_path / "contract.yaml")
    json_path = save_validation_report_json(
        clean_res, corrupted_res, tmp_path / "validation.json"
    )
    assert yaml_path.exists()
    assert json_path.exists()
    assert "dataContractSpecification" in yaml_path.read_text(encoding="utf-8")

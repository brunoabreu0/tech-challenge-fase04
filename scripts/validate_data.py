"""Script CLI da Etapa 1: Validação de Dados e Contratos com Pandera."""

from __future__ import annotations

import logging

from credit_guard.data.contracts import (
    DataContractValidator,
    export_data_contract_specification,
    generate_corrupted_batch,
    save_validation_report_json,
)
from credit_guard.data.loader import load_or_create_reference_dataset
from credit_guard.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Executa a validação no dataset limpo e demonstra o bloqueio de lote inválido."""
    settings = get_settings()
    validator = DataContractValidator(require_target=True)

    logger.info("Carregando Dataset de Referência limpo e sanitizado (LGPD)...")
    reference_df = load_or_create_reference_dataset()
    clean_result = validator.validate_batch(reference_df)
    logger.info(
        "Validação do Dataset de Referência (%d linhas): is_valid=%s | bloqueado=%s",
        clean_result.total_rows,
        clean_result.is_valid,
        clean_result.blocked_ingestion,
    )

    logger.info("Simulando lote corrompido para testar o Gatekeeper Pandera...")
    corrupted_df = generate_corrupted_batch(reference_df, n_samples=120, seed=42)
    corrupted_result = validator.validate_batch(corrupted_df)
    logger.warning(
        "Validação do Lote Corrompido (%d linhas): is_valid=%s | bloqueado=%s | "
        "linhas_com_erro=%d | violacoes=%d",
        corrupted_result.total_rows,
        corrupted_result.is_valid,
        corrupted_result.blocked_ingestion,
        corrupted_result.failed_rows_count,
        corrupted_result.violation_count,
    )
    logger.info(
        "Violações por coluna interceptadas: %s",
        corrupted_result.violations_by_column,
    )

    spec_path = export_data_contract_specification(
        settings.reports_dir / "data_contract_spec.yaml"
    )
    report_path = save_validation_report_json(
        clean_result,
        corrupted_result,
        settings.reports_dir / "data_contract_validation_report.json",
    )
    logger.info("Contrato DCS exportado em: %s", spec_path)
    logger.info("Relatório de validação salvo em: %s", report_path)


if __name__ == "__main__":
    main()

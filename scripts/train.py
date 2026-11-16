"""Script CLI de treinamento e avaliação do modelo base de risco de crédito."""

from __future__ import annotations

import json
import logging

from sklearn.model_selection import train_test_split

from credit_guard.data.contracts import DataContractValidator
from credit_guard.data.loader import (
    load_or_create_reference_dataset,
    split_features_and_target,
)
from credit_guard.model.factory import ClassifierFactory
from credit_guard.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Valida o Dataset de Referência, treina XGBoost e Logistic e salva métricas."""
    settings = get_settings()
    df = load_or_create_reference_dataset()

    validator = DataContractValidator(require_target=True)
    contract_res = validator.validate_batch(df)
    if not contract_res.is_valid:
        msg = f"Contrato de dados falhou com {contract_res.violation_count} erros."
        raise RuntimeError(msg)

    logger.info(
        "Dataset de Referência validado pelo Pandera (%d amostras). Treinando...",
        len(df),
    )

    x, y = split_features_and_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=settings.random_seed,
        stratify=y,
    )

    summary: dict[str, dict[str, float]] = {}
    for model_type in ("xgboost", "logistic"):
        clf = ClassifierFactory.create(model_type, random_state=settings.random_seed)
        clf.fit(x_train, y_train)
        metrics = clf.evaluate(x_test, y_test)
        summary[model_type] = metrics
        out_path = settings.model_dir / f"credit_{model_type}.joblib"
        clf.save(out_path)
        logger.info(
            "Modelo '%s' salvo em %s | Métricas: %s", model_type, out_path, metrics
        )

    metrics_path = settings.reports_dir / "reference_model_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Resumo de métricas de referência salvo em: %s", metrics_path)


if __name__ == "__main__":
    main()

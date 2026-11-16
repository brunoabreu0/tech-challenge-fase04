"""Fixtures compartilhadas para os testes automatizados."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_guard.data.loader import generate_credit_dataset
from credit_guard.data.privacy import PrivacyGuard


@pytest.fixture(scope="session")
def raw_credit_df() -> pd.DataFrame:
    """Dataset bruto com colunas PII para testes de privacidade e loader."""
    return generate_credit_dataset(n_samples=600, seed=42, include_pii=True)


@pytest.fixture(scope="session")
def reference_credit_df(raw_credit_df: pd.DataFrame) -> pd.DataFrame:
    """Dataset de referência sanitizado pelo PrivacyGuard."""
    guard = PrivacyGuard()
    return guard.sanitize_dataframe(raw_credit_df, drop_direct_pii=True)

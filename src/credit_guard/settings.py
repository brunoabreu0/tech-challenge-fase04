"""Configurações centralizadas da aplicação via pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações globais de ambiente, caminhos, drift e governança LGPD."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="development", description="Ambiente de execução")
    app_name: str = Field(
        default="credit-scoring-observability-api",
        description="Nome do serviço",
    )
    app_version: str = Field(default="0.1.0", description="Versão da aplicação")
    log_level: str = Field(default="INFO", description="Nível de log")
    random_seed: int = Field(default=42, description="Semente de reprodutibilidade")

    data_raw_dir: Path = Field(
        default=Path("data/raw"),
        description="Diretório de dados brutos",
    )
    data_processed_dir: Path = Field(
        default=Path("data/processed"),
        description="Diretório de dados processados",
    )
    model_dir: Path = Field(
        default=Path("models"),
        description="Diretório de modelos serializados",
    )
    reports_dir: Path = Field(
        default=Path("reports"),
        description="Diretório de relatórios HTML/JSON de drift e qualidade",
    )

    default_model_type: str = Field(
        default="xgboost",
        description="Tipo padrão de classificador (xgboost ou logistic)",
    )
    reference_samples: int = Field(
        default=6000,
        ge=5000,
        description="Quantidade mínima de amostras no dataset de referência (>= 5000)",
    )
    production_samples: int = Field(
        default=3000,
        ge=500,
        description="Quantidade de amostras no lote de produção para análise de drift",
    )

    psi_warning_threshold: float = Field(
        default=0.10,
        description="Limiar de alerta moderado para Population Stability Index (PSI)",
    )
    psi_critical_threshold: float = Field(
        default=0.25,
        description="Limiar crítico de drift severo para PSI",
    )
    ks_alpha: float = Field(
        default=0.05,
        description="Nível de significância (p-valor) para o teste Kolmogorov-Smirnov",
    )

    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="URI do servidor de rastreamento do MLflow",
    )
    mlflow_experiment_name: str = Field(
        default="credit-scoring-drift-monitoring",
        description="Nome do experimento no MLflow",
    )

    pii_hash_salt: str = Field(
        default="fiap-postech-9mlet-fase04-salt",
        description="Salt criptográfico para pseudonimização SHA-256 de PII (LGPD)",
    )
    data_retention_days: int = Field(
        default=1825,
        description="Prazo de retenção em dias (5 anos - BACEN/LGPD)",
    )


@lru_cache
def get_settings() -> Settings:
    """Retorna instância singleton das configurações da aplicação."""
    return Settings()

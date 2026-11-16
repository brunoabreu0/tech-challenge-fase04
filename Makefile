VENV = .venv
VENV_BIN = $(VENV)/bin
PYTHON = $(VENV_BIN)/python
PIP = $(VENV_BIN)/pip
PYTEST = $(VENV_BIN)/pytest
RUFF = $(VENV_BIN)/ruff
UVICORN = $(VENV_BIN)/uvicorn

.PHONY: venv install setup clean lint format test test-cov \
        download-data validate-data train drift-report mlflow-track pipeline-all \
        run-api docker-build docker-run compose-up compose-down compose-logs \
        tf-init tf-plan tf-apply tf-destroy

# --------------------------------------------------------------------------- #
# Ambiente local (Python 3.12 + Poetry)                                       #
# --------------------------------------------------------------------------- #
$(VENV)/bin/activate:
	/opt/homebrew/bin/python3.12 -m venv $(VENV) || python3.12 -m venv $(VENV)

venv: $(VENV)/bin/activate

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install poetry
	$(VENV_BIN)/poetry install --with dev

setup: install
	$(VENV_BIN)/pre-commit install
	cp -n .env.example .env || true

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true

# --------------------------------------------------------------------------- #
# Qualidade de código                                                         #
# --------------------------------------------------------------------------- #
lint:
	$(RUFF) check .

format:
	$(RUFF) format .

# --------------------------------------------------------------------------- #
# Testes                                                                      #
# --------------------------------------------------------------------------- #
test:
	$(PYTEST) tests/ -v

test-cov:
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# --------------------------------------------------------------------------- #
# Etapas do Tech Challenge Fase 4                                             #
# --------------------------------------------------------------------------- #
download-data:
	PYTHONPATH=src $(PYTHON) -c \
	  "from credit_guard.data.loader import download_kaggle_dataset; from pathlib import Path; \
	  ok = download_kaggle_dataset(Path('data/raw')); \
	  print('Kaggle download:', 'sucesso ✓' if ok else 'fallback determinístico local ativo ✓')"

validate-data:
	PYTHONPATH=src $(PYTHON) scripts/validate_data.py

train:
	PYTHONPATH=src $(PYTHON) scripts/train.py

drift-report:
	PYTHONPATH=src $(PYTHON) scripts/run_drift_analysis.py

mlflow-track:
	PYTHONPATH=src $(PYTHON) scripts/run_mlflow_tracking.py

pipeline-all: validate-data train drift-report mlflow-track

# --------------------------------------------------------------------------- #
# API                                                                         #
# --------------------------------------------------------------------------- #
run-api:
	PYTHONPATH=src $(UVICORN) credit_guard.api.main:app --reload --host 0.0.0.0 --port 8000

# --------------------------------------------------------------------------- #
# Docker & Docker Compose                                                     #
# --------------------------------------------------------------------------- #
docker-build:
	docker build -t credit-guard-api:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env credit-guard-api:latest

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f api

# --------------------------------------------------------------------------- #
# Terraform (AWS)                                                             #
# --------------------------------------------------------------------------- #
tf-init:
	cd terraform && terraform init

tf-plan:
	cd terraform && terraform plan

tf-apply:
	cd terraform && terraform apply -auto-approve

tf-destroy:
	cd terraform && terraform destroy -auto-approve

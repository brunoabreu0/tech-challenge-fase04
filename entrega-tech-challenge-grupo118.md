# Entrega — Tech Challenge (Fase 4: Model Monitoring and Security)

**Curso:** FIAP Pós-Tech — Machine Learning Engineering  
**Turma:** 9MLET  
**Grupo:** 118

**Autores:**
* **Bruno Machado Abreu** (RM372965)
* **Renan Prado Gonzalez** (RM374089)
* **Davi Coene Rosa** (RM371466)
* **Paulo Henrique Alves Krempel** (RM374144)
* **Pedro Gabriel Pereira do Nascimento** (RM372994)

---

## 🔗 Links Oficiais da Entrega

1. **Repositório GitHub:**
   * [https://github.com/brunoabreu0/tech-challenge-fase04](https://github.com/brunoabreu0/tech-challenge-fase04)
   * Contém o código-fonte completo em Python 3.12 (`src/credit_guard/`), contratos de dados com **Pandera**, motor de simulação e detecção de Drift com **Evidently AI** (PSI, KS, Wasserstein, JSD, CUSUM), rastreamento **MLflow**, API **FastAPI**, stack **Prometheus + Grafana** auto-provisionada, suíte de testes (**98% de cobertura**), pipeline CI/CD GitHub Actions e infraestrutura **Terraform AWS**.

2. **Apresentação em Vídeo (Método STAR — ≤ 5 min):**
   * **YouTube:** *(inserir link do vídeo após gravação)*

---

## ✅ Checklist de Atendimento aos Critérios de Avaliação da Banca

| Critério da Rubrica | Peso | Entrega Realizada no Projeto |
|---|:---:|---|
| **1. Validação de Dados (Contratos)** | **25%** | Implementado em `src/credit_guard/data/contracts.py` e `scripts/validate_data.py` com **Pandera (`lazy=True`)** contendo **11 regras rígidas** (`age > 18`, `annual_income` sem nulos e dentro de limites, `application_id` sem duplicatas, `credit_score` entre 300 e 850, consistência idade vs. tempo de emprego) + exportação **Data Contract Specification YAML** + bloqueio comprovado de lote corrompido (HTTP 422 na API e status `BLOCKED` no pipeline). |
| **2. Detecção de Drift** | **25%** | Implementado em `src/credit_guard/drift/simulator.py`, `src/credit_guard/drift/detector.py` e `scripts/run_drift_analysis.py`. Separação estrita entre Dataset de Referência (`6.000 amostras`) e Produção (`3.000 amostras`), simulação de degradação econômica em 6 variáveis + Concept Drift, cálculo de **PSI**, **Kolmogorov-Smirnov (KS)**, **Wasserstein**, **Jensen-Shannon** e **CUSUM**, e geração de relatórios HTML interativos do **Evidently AI**. |
| **3. Observabilidade e Logs** | **20%** | Implementado em `src/credit_guard/observability/`, `src/credit_guard/api/` e `monitoring/`. Logs estruturados JSONL centralizados (`pipeline_execution_events.jsonl`), rastreamento completo no **MLflow** (parâmetros, métricas de drift, model signature e artefatos HTML) e dashboard auto-provisionado no **Grafana** (11 painéis alimentados via **Prometheus**). |
| **4. Governança (LGPD) e Causalidade** | **15%** | Implementado em `src/credit_guard/data/privacy.py` (`PrivacyGuard`: pseudonimização SHA-256 com salt, mascaramento de CPF, generalização de CEP para k-anonimato e auditoria de Fairness etário) + documentação completa no `README.md` cobrindo Base Legal (Art. 7º X e Art. 20º da LGPD), Plano de Retenção de 5 anos (1.825 dias) e Grafo Causal (DAG) da degradação silenciosa. |
| **5. Vídeo STAR** | **15%** | Roteiro estruturado no `README.md` cobrindo *Situation*, *Task*, *Action* e *Result* em até 5 minutos. |

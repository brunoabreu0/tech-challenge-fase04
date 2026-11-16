"""Motor estatístico de detecção de Data/Concept Drift (PSI, KS, Evidently)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from credit_guard.data.loader import split_features_and_target
from credit_guard.data.preprocessor import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from credit_guard.model.base import BaseClassifier
from credit_guard.settings import get_settings


class FeatureDriftMetric(BaseModel):
    """Métricas estatísticas de Drift para uma variável individual."""

    model_config = ConfigDict(frozen=True)

    feature: str
    feature_type: str
    psi: float = Field(description="Population Stability Index (PSI)")
    ks_statistic: float | None = Field(
        default=None, description="Estatística do teste Kolmogorov-Smirnov"
    )
    ks_p_value: float | None = Field(
        default=None, description="P-valor do teste Kolmogorov-Smirnov"
    )
    wasserstein_norm: float | None = Field(
        default=None,
        description="Distância de Wasserstein normalizada ( desvios-padrão)",
    )
    js_divergence: float = Field(description="Divergência de Jensen-Shannon")
    ref_mean: float | None = None
    prod_mean: float | None = None
    relative_shift_pct: float | None = None
    drift_detected: bool
    severity: str = Field(description="STABLE, WARNING ou CRITICAL")


class ConceptDriftSummary(BaseModel):
    """Diagnóstico de degradação de desempenho e Concept Drift em produção."""

    model_config = ConfigDict(frozen=True)

    reference_roc_auc: float
    production_roc_auc: float
    roc_auc_drop: float
    reference_f1: float
    production_f1: float
    reference_brier: float
    production_brier: float
    reference_default_rate: float
    production_actual_default_rate: float
    production_predicted_default_rate: float
    unpredicted_risk_gap: float
    cusum_alert_triggered: bool
    cusum_max_score: float
    concept_drift_detected: bool


class DriftAnalysisSummary(BaseModel):
    """Relatório estruturado completo de Data Drift, Label Drift e Concept Drift."""

    model_config = ConfigDict(frozen=True)

    scenario_name: str
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    reference_rows: int
    production_rows: int
    dataset_drift_detected: bool
    drifted_features_count: int
    total_features_evaluated: int
    share_of_drifted_features: float
    max_psi: float
    mean_psi: float
    overall_severity: str
    drifted_feature_names: list[str]
    feature_metrics: list[FeatureDriftMetric]
    prediction_psi: float | None = None
    concept_drift: ConceptDriftSummary | None = None
    evidently_html_path: str | None = None


def compute_psi_numeric(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Calcula o Population Stability Index (PSI) para variáveis contínuas/discretas."""
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]
    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    ref_pct = np.clip(ref_counts / len(ref_clean), eps, 1.0)
    cur_pct = np.clip(cur_counts / len(cur_clean), eps, 1.0)

    psi_val = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(max(psi_val, 0.0), 4)


def compute_psi_categorical(
    reference: pd.Series,
    current: pd.Series,
    eps: float = 1e-4,
) -> float:
    """Calcula o PSI para variáveis categóricas com base nas frequências relativas."""
    categories = sorted(
        set(reference.dropna().astype(str)).union(set(current.dropna().astype(str)))
    )
    if not categories:
        return 0.0

    ref_freq = reference.astype(str).value_counts(normalize=True)
    cur_freq = current.astype(str).value_counts(normalize=True)

    psi_total = 0.0
    for cat in categories:
        p = max(float(ref_freq.get(cat, 0.0)), eps)
        q = max(float(cur_freq.get(cat, 0.0)), eps)
        psi_total += (q - p) * float(np.log(q / p))
    return round(max(psi_total, 0.0), 4)


def compute_js_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 12,
    eps: float = 1e-6,
) -> float:
    """Calcula a Divergência de Jensen-Shannon (simétrica e limitada em [0, ln(2)])."""
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]
    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    low = float(min(np.min(ref_clean), np.min(cur_clean)))
    high = float(max(np.max(ref_clean), np.max(cur_clean)))
    if low == high:
        return 0.0

    bins = np.linspace(low, high, n_bins + 1)
    p_hist, _ = np.histogram(ref_clean, bins=bins)
    q_hist, _ = np.histogram(cur_clean, bins=bins)

    p = np.clip(p_hist / max(np.sum(p_hist), 1), eps, 1.0)
    q = np.clip(q_hist / max(np.sum(q_hist), 1), eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)

    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return round(max(0.5 * (kl_pm + kl_qm), 0.0), 4)


def detect_cusum_regime_shift(
    residuals: np.ndarray,
    drift_slack: float = 0.03,
    threshold: float = 8.0,
) -> tuple[bool, float]:
    """Executa o algoritmo CUSUM unilateral para detectar mudança de regime nos erros.

    Monitora o resíduo entre inadimplência observada (`y_true`) e probabilidade
    prevista (`y_prob`), sinalizando quando o modelo subestima sistematicamente o risco.
    """
    s_pos = 0.0
    max_s = 0.0
    for err in residuals:
        s_pos = max(0.0, s_pos + float(err) - drift_slack)
        if s_pos > max_s:
            max_s = s_pos
    return max_s >= threshold, round(max_s, 4)


class DriftDetector:
    """Orquestra testes estatísticos (PSI, KS, Wasserstein, CUSUM) e Evidently."""

    def __init__(
        self,
        psi_warning: float | None = None,
        psi_critical: float | None = None,
        ks_alpha: float | None = None,
    ) -> None:
        """Configura limiares estatísticos a partir do `Settings`."""
        settings = get_settings()
        self.psi_warning = (
            psi_warning if psi_warning is not None else settings.psi_warning_threshold
        )
        self.psi_critical = (
            psi_critical
            if psi_critical is not None
            else settings.psi_critical_threshold
        )
        self.ks_alpha = ks_alpha if ks_alpha is not None else settings.ks_alpha

    def _classify_severity(
        self, psi: float, ks_p_value: float | None
    ) -> tuple[bool, str]:
        """Determina severidade combinando regra de PSI e significância do KS."""
        if psi >= self.psi_critical:
            return True, "CRITICAL"
        if psi >= self.psi_warning or (
            ks_p_value is not None and ks_p_value < self.ks_alpha and psi >= 0.05
        ):
            return True, "WARNING"
        return False, "STABLE"

    def analyze(
        self,
        reference_df: pd.DataFrame,
        production_df: pd.DataFrame,
        classifier: BaseClassifier | None = None,
        scenario_name: str = "severe_stagflation",
        evidently_html_path: str | None = None,
    ) -> DriftAnalysisSummary:
        """Executa bateria de testes de Data e Concept Drift (Ref vs Prod)."""
        metrics_list: list[FeatureDriftMetric] = []
        drifted_names: list[str] = []

        for col in NUMERIC_FEATURES:
            ref_arr = reference_df[col].to_numpy(dtype=float)
            prod_arr = production_df[col].to_numpy(dtype=float)

            psi = compute_psi_numeric(ref_arr, prod_arr)
            ks_res = stats.ks_2samp(ref_arr, prod_arr)
            ks_stat = round(float(ks_res.statistic), 4)
            ks_pval = round(float(ks_res.pvalue), 6)

            ref_std = max(float(np.std(ref_arr)), 1e-6)
            w_dist = round(
                float(stats.wasserstein_distance(ref_arr, prod_arr)) / ref_std, 4
            )
            js_div = compute_js_divergence(ref_arr, prod_arr)

            ref_mean = round(float(np.mean(ref_arr)), 4)
            prod_mean = round(float(np.mean(prod_arr)), 4)
            rel_shift = (
                round(((prod_mean - ref_mean) / abs(ref_mean)) * 100.0, 2)
                if abs(ref_mean) > 1e-6
                else 0.0
            )

            is_drifted, severity = self._classify_severity(psi, ks_pval)
            if is_drifted:
                drifted_names.append(col)

            metrics_list.append(
                FeatureDriftMetric(
                    feature=col,
                    feature_type="numeric",
                    psi=psi,
                    ks_statistic=ks_stat,
                    ks_p_value=ks_pval,
                    wasserstein_norm=w_dist,
                    js_divergence=js_div,
                    ref_mean=ref_mean,
                    prod_mean=prod_mean,
                    relative_shift_pct=rel_shift,
                    drift_detected=is_drifted,
                    severity=severity,
                )
            )

        for col in CATEGORICAL_FEATURES:
            psi = compute_psi_categorical(reference_df[col], production_df[col])
            is_drifted, severity = self._classify_severity(psi, None)
            if is_drifted:
                drifted_names.append(col)

            metrics_list.append(
                FeatureDriftMetric(
                    feature=col,
                    feature_type="categorical",
                    psi=psi,
                    ks_statistic=None,
                    ks_p_value=None,
                    wasserstein_norm=None,
                    js_divergence=round(psi * 0.25, 4),
                    ref_mean=None,
                    prod_mean=None,
                    relative_shift_pct=None,
                    drift_detected=is_drifted,
                    severity=severity,
                )
            )

        psi_values = [m.psi for m in metrics_list]
        max_psi = round(max(psi_values), 4) if psi_values else 0.0
        mean_psi = round(float(np.mean(psi_values)), 4) if psi_values else 0.0

        prediction_psi: float | None = None
        concept_summary: ConceptDriftSummary | None = None

        if classifier is not None and TARGET_COLUMN in production_df.columns:
            x_ref, y_ref = split_features_and_target(reference_df)
            x_prod, y_prod = split_features_and_target(production_df)

            ref_eval = classifier.evaluate(x_ref, y_ref)
            prod_eval = classifier.evaluate(x_prod, y_prod)

            ref_probs = classifier.predict_proba(x_ref)
            prod_probs = classifier.predict_proba(x_prod)
            prediction_psi = compute_psi_numeric(ref_probs, prod_probs)

            residuals = y_prod.to_numpy(dtype=float) - prod_probs
            cusum_triggered, cusum_score = detect_cusum_regime_shift(residuals)

            auc_drop = round(ref_eval["roc_auc"] - prod_eval["roc_auc"], 4)
            risk_gap = round(
                prod_eval["actual_default_rate"]
                - prod_eval["mean_predicted_probability"],
                4,
            )
            concept_detected = bool(
                cusum_triggered or auc_drop >= 0.03 or abs(risk_gap) >= 0.05
            )

            concept_summary = ConceptDriftSummary(
                reference_roc_auc=ref_eval["roc_auc"],
                production_roc_auc=prod_eval["roc_auc"],
                roc_auc_drop=auc_drop,
                reference_f1=ref_eval["f1_score"],
                production_f1=prod_eval["f1_score"],
                reference_brier=ref_eval["brier_score"],
                production_brier=prod_eval["brier_score"],
                reference_default_rate=ref_eval["actual_default_rate"],
                production_actual_default_rate=prod_eval["actual_default_rate"],
                production_predicted_default_rate=prod_eval[
                    "mean_predicted_probability"
                ],
                unpredicted_risk_gap=risk_gap,
                cusum_alert_triggered=cusum_triggered,
                cusum_max_score=cusum_score,
                concept_drift_detected=concept_detected,
            )

        if max_psi >= self.psi_critical or (
            concept_summary is not None and concept_summary.concept_drift_detected
        ):
            overall_severity = "CRITICAL"
        elif len(drifted_names) > 0:
            overall_severity = "WARNING"
        else:
            overall_severity = "STABLE"

        share = round(len(drifted_names) / max(len(metrics_list), 1), 4)
        return DriftAnalysisSummary(
            scenario_name=scenario_name,
            reference_rows=len(reference_df),
            production_rows=len(production_df),
            dataset_drift_detected=len(drifted_names) >= 2,
            drifted_features_count=len(drifted_names),
            total_features_evaluated=len(metrics_list),
            share_of_drifted_features=share,
            max_psi=max_psi,
            mean_psi=mean_psi,
            overall_severity=overall_severity,
            drifted_feature_names=drifted_names,
            feature_metrics=metrics_list,
            prediction_psi=prediction_psi,
            concept_drift=concept_summary,
            evidently_html_path=evidently_html_path,
        )

    def generate_evidently_reports(
        self,
        reference_df: pd.DataFrame,
        production_df: pd.DataFrame,
        classifier: BaseClassifier | None = None,
        scenario_name: str = "severe_stagflation",
        output_dir: Path | None = None,
    ) -> tuple[DriftAnalysisSummary, Path, Path]:
        """Gera o relatório HTML oficial do Evidently AI e o JSON estatístico."""
        settings = get_settings()
        target_dir = output_dir or settings.reports_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        cols_for_evidently = list(FEATURE_COLUMNS)
        if (
            TARGET_COLUMN in reference_df.columns
            and TARGET_COLUMN in production_df.columns
        ):
            cols_for_evidently.append(TARGET_COLUMN)

        ref_sub = reference_df[cols_for_evidently].copy()
        prod_sub = production_df[cols_for_evidently].copy()

        if classifier is not None:
            ref_sub["predicted_probability"] = np.round(
                classifier.predict_proba(reference_df), 4
            )
            prod_sub["predicted_probability"] = np.round(
                classifier.predict_proba(production_df), 4
            )

        evidently_report = Report([DataDriftPreset()])
        snapshot = evidently_report.run(
            current_data=prod_sub,
            reference_data=ref_sub,
        )

        html_path = target_dir / f"evidently_drift_{scenario_name}.html"
        evidently_json_path = target_dir / f"evidently_snapshot_{scenario_name}.json"
        snapshot.save_html(str(html_path))
        snapshot.save_json(str(evidently_json_path))

        summary = self.analyze(
            reference_df=reference_df,
            production_df=production_df,
            classifier=classifier,
            scenario_name=scenario_name,
            evidently_html_path=str(html_path),
        )

        summary_json_path = target_dir / f"drift_summary_{scenario_name}.json"
        summary_json_path.write_text(
            json.dumps(summary.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary, html_path, summary_json_path


def render_executive_drift_html(
    summary: DriftAnalysisSummary, output_path: Path
) -> Path:
    """Gera painel executivo HTML complementar com PSI, KS e Concept Drift."""
    rows_html = []
    for m in sorted(summary.feature_metrics, key=lambda x: -x.psi):
        badge_color = (
            "#dc2626"
            if m.severity == "CRITICAL"
            else ("#d97706" if m.severity == "WARNING" else "#16a34a")
        )
        ks_str = (
            f"{m.ks_statistic:.4f} (p={m.ks_p_value:.4g})"
            if m.ks_statistic is not None
            else "N/A"
        )
        w_str = f"{m.wasserstein_norm:.4f}" if m.wasserstein_norm is not None else "N/A"
        shift_str = (
            f"{m.relative_shift_pct:+.2f}% ({m.ref_mean} → {m.prod_mean})"
            if m.relative_shift_pct is not None
            else "Categórica"
        )
        rows_html.append(
            f"<tr>"
            f"<td><strong>{m.feature}</strong></td>"
            f"<td>{m.feature_type}</td>"
            f"<td><strong>{m.psi:.4f}</strong></td>"
            f"<td>{ks_str}</td>"
            f"<td>{w_str}</td>"
            f"<td>{m.js_divergence:.4f}</td>"
            f"<td>{shift_str}</td>"
            f"<td><span style='background:{badge_color};color:#fff;padding:3px 8px;"
            f"border-radius:4px;font-size:12px;'>{m.severity}</span></td>"
            f"</tr>"
        )

    cd_html = ""
    if summary.concept_drift is not None:
        cd = summary.concept_drift
        cd_html = (
            f"<div class='card'>"
            f"<h2>Diagnóstico de Concept Drift e Degradação Silenciosa</h2>"
            f"<p><strong>ROC-AUC (Ref vs Prod):</strong> "
            f"{cd.reference_roc_auc:.4f} → "
            f"{cd.production_roc_auc:.4f} (Queda: {cd.roc_auc_drop:+.4f})</p>"
            f"<p><strong>Brier Score (Calibração):</strong> "
            f"{cd.reference_brier:.4f} → {cd.production_brier:.4f}</p>"
            f"<p><strong>Inadimplência Real vs Prevista em Produção:</strong> "
            f"{cd.production_actual_default_rate:.2%} real vs "
            f"{cd.production_predicted_default_rate:.2%} prevista "
            f"(Gap subestimado: <strong>{cd.unpredicted_risk_gap:+.2%}</strong>)</p>"
            f"<p><strong>Detector de Mudança de Regime CUSUM:</strong> "
            f"Score={cd.cusum_max_score:.2f} | Alerta={cd.cusum_alert_triggered}</p>"
            f"</div>"
        )

    ref_n = summary.reference_rows
    prod_n = summary.production_rows
    drifted_cnt = summary.drifted_features_count
    total_cnt = summary.total_features_evaluated
    drifted_list = ", ".join(summary.drifted_feature_names)
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"/>
<title>Painel de Observabilidade e Drift — Credit Scoring</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #0f172a; color: #f8fafc; margin: 24px; }}
.card {{ background: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 20px;
         border: 1px solid #334155; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
th, td {{ padding: 10px 12px; border-bottom: 1px solid #334155; text-align: left; }}
th {{ background: #0f172a; color: #94a3b8; }}
</style>
</head>
<body>
<div class="card">
  <h1>Relatório Estatístico de Data Drift & Concept Drift ({summary.scenario_name})</h1>
  <p>Referência: {ref_n} amostras | Produção: {prod_n} amostras |
     Severidade Global: <strong>{summary.overall_severity}</strong> |
     Features com Drift: <strong>{drifted_cnt}/{total_cnt}</strong>
     ({drifted_list})</p>
</div>
{cd_html}
<div class="card">
  <h2>Testes Estatísticos por Variável (PSI, Kolmogorov-Smirnov, Wasserstein, JSD)</h2>
  <table>
    <thead>
      <tr>
        <th>Feature</th><th>Tipo</th><th>PSI</th><th>Teste KS (p-valor)</th>
        <th>Wasserstein Norm</th><th>JSD</th><th>Deslocamento Médio</th><th>Status</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path

import { api } from '../services/api.js';
import { horizontalBars, lineChart } from '../components/charts.js';

const value = (number, digits = 2) => {
  if (number === null || number === undefined || number === '' || Number.isNaN(Number(number))) return 'N/A';
  return Number(number).toFixed(digits);
};
const metric = (number, digits = 2, suffix = '') => `${value(number, digits)}${value(number, digits) === 'N/A' ? '' : suffix}`;
const escape = (text = '') => String(text).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
function errorBuckets(rows, key) {
  return [[0,5],[5,15],[15,30],[30,Infinity]].map(([from,to]) => ({
    label: `${from}-${to === Infinity ? '∞' : to}`,
    value: rows.filter(r => { const e=Math.abs(Number(r[key])); return e>=from && e<to; }).length,
  }));
}

export async function PredictionAccuracyPage(root) {
  let selected = api.getValidationModel();

  async function render() {
  const [report, validation, rolling] = await Promise.all([api.validationReport(selected), api.predictionValidation(100, selected), api.rollingValidation(selected)]);
  const rows = validation.items;
  const lifecycle = report.model_family === 'monthly_lifecycle';
  const risk = report.risk_model || report.risk_classification;
  const delayMae = report.delay_model.MAE_days ?? report.delay_model.MAE;
  const delayRmse = report.delay_model.RMSE_days ?? report.delay_model.RMSE;
  const confidenceValues = rows.map(row => Number(row.model_confidence_percentage)).filter(Number.isFinite);
  const averageConfidence = confidenceValues.length ? confidenceValues.reduce((total, item) => total + item, 0) / confidenceValues.length : null;
  const quality = report.metadata.feature_quality || {};
  const calibration = report.confidence_calibration || {};
  const sectors = Object.entries(report.sector_validation?.sectors || {});
  const shapValidation = report.metadata.shap_validation || {};
  const shapTargets = lifecycle ? Object.entries(report.shap || {}) : Object.entries(shapValidation.targets || {});
  root.innerHTML = `<header class="page-head"><div><span class="kicker">Historical cutoff backtest · ${report.model_version || validation.model_version || 'legacy'}</span><h1>Prediction Accuracy Dashboard</h1><p>Metrics and validation rows are generated from the active selected training window, using only later completed projects for evaluation.</p></div></header>
    <section class="panel"><div class="panel-head"><div><span class="kicker">Model selection</span><h2>Load a retrained year range</h2></div><div class="filters compact-filters"><input id="validation-model" class="data-input" value="${selected}" placeholder="Example: 2001_2015"/><button class="secondary-btn" id="load-validation">Load model</button></div></div><p class="muted">Leave blank to use the currently active model. Enter a model version created by retraining, such as <code>2001_2015</code>.</p></section>
    <div class="notice"><strong>Model family:</strong> ${lifecycle ? 'Monthly Lifecycle' : 'Legacy verification'} · <strong>Training:</strong> ${report.metadata.training_start ?? 'N/A'}-${report.metadata.training_end ?? 'N/A'} · <strong>Future holdout:</strong> ${report.metadata.evaluated_test_start ?? 'N/A'}-${report.metadata.evaluated_test_end ?? 'N/A'}. ${report.metadata.leakage_policy || report.metadata.future_information_policy || ''}</div>
    <div class="stat-grid"><article class="stat-card blue"><span class="stat-eyebrow">Cost MAE</span><strong class="stat-value">${metric(report.cost_model.MAE)} pp</strong><small class="stat-note">RMSE ${metric(report.cost_model.RMSE)} · MAPE ${metric(report.cost_model.MAPE)}% · R² ${value(report.cost_model.R2, 3)}</small></article><article class="stat-card amber"><span class="stat-eyebrow">Delay MAE</span><strong class="stat-value">${metric(delayMae)} days</strong><small class="stat-note">RMSE ${metric(delayRmse)} · ${lifecycle ? `MAPE ${metric(report.delay_model.MAPE)}%` : `log-target RMSE ${value(report.delay_model.log_target_RMSE, 4)}`} · R² ${value(report.delay_model.R2, 3)}</small></article><article class="stat-card green"><span class="stat-eyebrow">${lifecycle ? 'Risk macro-F1' : 'Risk classification'}</span><strong class="stat-value">${value((lifecycle ? risk.macro_f1 : risk.accuracy) * 100, 1)}%</strong><small class="stat-note">${lifecycle ? `Accuracy ${metric(risk.accuracy * 100, 1, '%')} · macro precision ${metric(risk.macro_precision * 100, 1, '%')} · macro recall ${metric(risk.macro_recall * 100, 1, '%')}` : `Precision ${metric(risk.precision * 100, 1, '%')} · recall ${metric(risk.recall * 100, 1, '%')} · F1 ${metric(risk.f1 * 100, 1, '%')}`}</small></article><article class="stat-card"><span class="stat-eyebrow">Model confidence</span><strong class="stat-value">${metric(averageConfidence, 1, '%')}</strong><small class="stat-note">${lifecycle ? 'Not generated for this lifecycle run' : 'Earlier-year calibrated interval coverage; warning shown when final holdout drifts'}</small></article></div>
    <div class="stat-grid"><article class="stat-card blue"><span class="stat-eyebrow">Features used</span><strong class="stat-value">${value(report.metadata.feature_count, 0)}</strong><small class="stat-note">Only audited official-data features</small></article><article class="stat-card amber"><span class="stat-eyebrow">Removed invalid</span><strong class="stat-value">${value(quality.removed_invalid_feature_count, 0)}</strong><small class="stat-note">Empty, constant, synthetic, or unavailable fields</small></article><article class="stat-card green"><span class="stat-eyebrow">Data quality</span><strong class="stat-value">${metric(quality.data_quality_score, 1, '%')}</strong><small class="stat-note">Availability across retained features</small></article><article class="stat-card"><span class="stat-eyebrow">Confidence calibration</span><strong class="stat-value">${escape(calibration.status || (lifecycle ? 'not generated for this run' : 'unavailable')).replaceAll('_', ' ')}</strong><small class="stat-note">Cost coverage ${metric(calibration.holdout_observed?.cost_interval_coverage_percentage, 1, '%')} · delay ${metric(calibration.holdout_observed?.delay_interval_coverage_percentage, 1, '%')}</small></article></div>
    <div class="model-grid"><section class="panel"><div class="panel-head"><div><span class="kicker">${lifecycle ? 'Not generated for this lifecycle run' : 'Validation-safe sector breakdown'}</span><h2>Sector performance</h2></div></div><div class="table-wrap">${sectors.length ? `<table class="data-table compact-table"><thead><tr><th>Sector</th><th>Projects</th><th>Cost MAE</th><th>Delay MAE</th></tr></thead><tbody>${sectors.map(([name, item]) => `<tr><td>${escape(name)}</td><td>${item.projects}</td><td>${metric(item.cost_mae)} pp</td><td>${metric(item.delay_mae)} days</td></tr>`).join('')}</tbody></table>` : '<p class="muted">Not generated for this run.</p>'}</div></section><section class="panel"><span class="kicker">${lifecycle ? 'Lifecycle feature importance' : 'Validated explanations'}</span><h2>${lifecycle ? 'SHAP / feature importance' : 'SHAP quality'}</h2>${lifecycle ? shapTargets.map(([target, item]) => `<p><strong>${escape(target)}:</strong> ${escape((item.features || []).slice(0, 5).map(feature => feature.feature).join(', ') || 'Not generated for this run')}</p>`).join('') || '<p class="muted">Not generated for this run.</p>' : `<div class="notice compact"><strong>${shapValidation.validated ? 'Validated' : 'Review warning'}:</strong> explanation factors are checked against the retained numerical and historical feature contract.</div>${shapTargets.map(([target, item]) => `<p><strong>${escape(target)}:</strong> ${escape((item.meaningful_expected_factors || []).join(', ') || item.status)}</p>`).join('')}`}</section></div>
    <section class="panel"><div class="panel-head"><div><span class="kicker">Expanding-window temporal validation · ${value(rolling.fold_count, 0)} folds</span><h2>Reliability across unseen completion years</h2></div></div>${rolling.folds?.length ? `<div class="model-grid"><div><h3>Cost MAE by test year</h3>${lineChart(rolling.folds.map((fold) => ({ label: fold.test_year, value: fold.cost_MAE })), { suffix: ' pp' })}</div><div><h3>Delay MAE by test year</h3>${lineChart(rolling.folds.map((fold) => ({ label: fold.test_year, value: fold.delay_MAE_days })), { suffix: ' days' })}</div></div><div class="notice compact"><strong>Average risk macro F1:</strong> ${metric(rolling.average_risk_f1 * 100, 1, '%')}. Every fold trains only on years before its displayed test year.</div>` : '<div class="notice compact">Rolling validation not generated for this lifecycle run.</div>'}</section>
    <div class="model-grid"><section class="panel"><div class="panel-head"><div><span class="kicker">Cost escalation</span><h2>Predicted vs actual</h2></div></div><h3>AI predicted</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.predicted_cost_overrun })), { suffix: '%' })}<h3>Actual final outcome</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.actual_cost_overrun })), { suffix: '%' })}</section><section class="panel"><div class="panel-head"><div><span class="kicker">Schedule extension</span><h2>Predicted vs actual</h2></div></div><h3>AI predicted</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.predicted_delay_days })), { suffix: ' days' })}<h3>Actual final outcome</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.actual_delay_days })), { suffix: ' days' })}</section></div>
    <div class="model-grid"><section class="panel"><div class="panel-head"><div><span class="kicker">Cost error</span><h2>Error distribution (percentage points)</h2></div></div>${horizontalBars(errorBuckets(rows, 'cost_error'), { format: (count) => `${count} projects` })}</section><section class="panel"><div class="panel-head"><div><span class="kicker">Delay error</span><h2>Error distribution (days)</h2></div></div>${horizontalBars(errorBuckets(rows, 'delay_error'), { format: (count) => `${count} projects` })}</section></div>
    <section class="panel"><div class="panel-head"><div><span class="kicker">${validation.total} completed projects</span><h2>Project-wise forecast accuracy and uncertainty</h2></div><div class="filters compact-filters"><label>Error metric<select id="validation-sort-metric"><option value="cost_error">Cost error</option><option value="delay_error">Delay error</option></select></label><button class="secondary-btn" id="validation-sort-low">Lowest error first</button><button class="secondary-btn" id="validation-sort-high">Highest error first</button></div></div><div class="table-wrap"><table class="data-table"><thead><tr><th>Project ID</th><th>Project name</th><th>Predicted cost</th><th>Cost P10–P90</th><th>Actual cost</th><th>Cost error</th><th>Predicted delay</th><th>Delay P10–P90</th><th>Actual delay</th><th>Delay error</th></tr></thead><tbody id="validation-rows"></tbody></table></div></section>`;
  const tableBody = root.querySelector('#validation-rows');
  const metric = root.querySelector('#validation-sort-metric');
  const renderRows = (sortKey = 'cost_error', direction = 'asc') => {
    const sorted = [...rows].sort((a, b) => {
      const left = Math.abs(Number(a[sortKey] || 0));
      const right = Math.abs(Number(b[sortKey] || 0));
      return direction === 'asc' ? left - right : right - left;
    });
    tableBody.innerHTML = sorted.map((row) => `<tr><td>${row.project_id || 'Not published'}</td><td>${row.project_name || 'Not reported'}</td><td>${metric(row.predicted_cost_overrun, 2, '%')}</td><td>${row.predicted_cost_p10 == null || row.predicted_cost_p90 == null ? 'N/A' : `${metric(row.predicted_cost_p10, 2, '%')}–${metric(row.predicted_cost_p90, 2, '%')}`}</td><td>${metric(row.actual_cost_overrun, 2, '%')}</td><td>${metric(row.cost_error)} pp</td><td>${metric(row.predicted_delay_days, 2, ' days')}</td><td>${row.predicted_delay_p10 == null || row.predicted_delay_p90 == null ? 'N/A' : `${value(row.predicted_delay_p10)}–${value(row.predicted_delay_p90)} days`}</td><td>${metric(row.actual_delay_days, 2, ' days')}</td><td>${metric(row.delay_error, 2, ' days')}</td></tr>`).join('');
  };
  renderRows();
  root.querySelector('#validation-sort-low').addEventListener('click', () => renderRows(metric.value, 'asc'));
  root.querySelector('#validation-sort-high').addEventListener('click', () => renderRows(metric.value, 'desc'));
  root.querySelector('#load-validation').addEventListener('click', () => {
    selected = root.querySelector('#validation-model').value.trim();
    api.setValidationModel(selected);
    render();
  });
  }

  await render();
}

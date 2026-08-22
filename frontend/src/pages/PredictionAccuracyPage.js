import { api } from '../services/api.js';
import { horizontalBars, lineChart } from '../components/charts.js';

const value = (number, digits = 2) => Number(number || 0).toFixed(digits);

function errorBuckets(rows, key) {
  return [[0,5],[5,15],[15,30],[30,Infinity]].map(([from,to]) => ({
    label: `${from}-${to === Infinity ? '∞' : to}`,
    value: rows.filter(r => { const e=Math.abs(Number(r[key])); return e>=from && e<to; }).length,
  }));
}

export async function PredictionAccuracyPage(root) {
  let selected = api.getValidationModel();

  async function render() {
  const [report, validation] = await Promise.all([api.validationReport(selected), api.predictionValidation(100, selected)]);
  const rows = validation.items;
  const risk = report.risk_model || report.risk_classification;
  const delayMae = report.delay_model.MAE_days ?? report.delay_model.MAE;
  const delayRmse = report.delay_model.RMSE_days ?? report.delay_model.RMSE;
  const averageConfidence = rows.reduce((total, row) => total + Number(row.model_confidence_percentage || 0), 0) / Math.max(rows.length, 1);
  root.innerHTML = `<header class="page-head"><div><span class="kicker">Historical cutoff backtest · ${report.model_version || validation.model_version || 'legacy'}</span><h1>Prediction Accuracy Dashboard</h1><p>Metrics and validation rows are generated from the active selected training window, using only later completed projects for evaluation.</p></div></header>
    <section class="panel"><div class="panel-head"><div><span class="kicker">Model selection</span><h2>Load a retrained year range</h2></div><div class="filters compact-filters"><input id="validation-model" class="data-input" value="${selected}" placeholder="Example: 2001_2015"/><button class="secondary-btn" id="load-validation">Load model</button></div></div><p class="muted">Leave blank to use the currently active model. Enter a model version created by retraining, such as <code>2001_2015</code>.</p></section>
    <div class="notice"><strong>Active model:</strong> ${report.model_version || validation.model_version || 'legacy'} · Training ${report.metadata.training_start ?? 'n/a'}-${report.metadata.training_end ?? 'n/a'} · evaluated test projects ${report.metadata.evaluated_test_start ?? 'n/a'}-${report.metadata.evaluated_test_end ?? 'n/a'}. ${report.metadata.leakage_policy || report.metadata.future_information_policy || ''}</div>
    <div class="stat-grid"><article class="stat-card blue"><span class="stat-eyebrow">Cost MAE</span><strong class="stat-value">${value(report.cost_model.MAE)} pp</strong><small class="stat-note">RMSE ${value(report.cost_model.RMSE)} · MAPE ${value(report.cost_model.MAPE)}% · accuracy ${value(report.cost_model.accuracy_percentage, 1)}%</small></article><article class="stat-card amber"><span class="stat-eyebrow">Delay MAE</span><strong class="stat-value">${value(delayMae)} days</strong><small class="stat-note">RMSE ${value(delayRmse)} · log-target RMSE ${value(report.delay_model.log_target_RMSE, 4)} · accuracy ${value(report.delay_model.accuracy_percentage, 1)}%</small></article><article class="stat-card green"><span class="stat-eyebrow">Risk classification</span><strong class="stat-value">${value(risk.accuracy * 100, 1)}%</strong><small class="stat-note">Precision ${value(risk.precision * 100, 1)}% · recall ${value(risk.recall * 100, 1)}% · F1 ${value(risk.f1 * 100, 1)}%</small></article><article class="stat-card"><span class="stat-eyebrow">Validation projects</span><strong class="stat-value">${validation.total}</strong><small class="stat-note">Official unseen completed projects</small></article></div>
    <div class="model-grid"><section class="panel"><div class="panel-head"><div><span class="kicker">Cost escalation</span><h2>Predicted vs actual</h2></div></div><h3>AI predicted</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.predicted_cost_overrun })), { suffix: '%' })}<h3>Actual final outcome</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.actual_cost_overrun })), { suffix: '%' })}</section><section class="panel"><div class="panel-head"><div><span class="kicker">Schedule extension</span><h2>Predicted vs actual</h2></div></div><h3>AI predicted</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.predicted_delay_days })), { suffix: ' days' })}<h3>Actual final outcome</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.actual_delay_days })), { suffix: ' days' })}</section></div>
    <div class="model-grid"><section class="panel"><div class="panel-head"><div><span class="kicker">Cost error</span><h2>Error distribution (percentage points)</h2></div></div>${horizontalBars(errorBuckets(rows, 'cost_error'), { format: (count) => `${count} projects` })}</section><section class="panel"><div class="panel-head"><div><span class="kicker">Delay error</span><h2>Error distribution (days)</h2></div></div>${horizontalBars(errorBuckets(rows, 'delay_error'), { format: (count) => `${count} projects` })}</section></div>
    <section class="panel"><div class="panel-head"><div><span class="kicker">${validation.total} completed projects</span><h2>Project-wise forecast accuracy</h2></div><div class="filters compact-filters"><label>Error metric<select id="validation-sort-metric"><option value="cost_error">Cost error</option><option value="delay_error">Delay error</option></select></label><button class="secondary-btn" id="validation-sort-low">Lowest error first</button><button class="secondary-btn" id="validation-sort-high">Highest error first</button></div></div><div class="table-wrap"><table class="data-table"><thead><tr><th>Project ID</th><th>Project name</th><th>Predicted cost</th><th>Actual cost</th><th>Cost error</th><th>Predicted delay</th><th>Actual delay</th><th>Delay error</th></tr></thead><tbody id="validation-rows"></tbody></table></div></section>`;
  const tableBody = root.querySelector('#validation-rows');
  const metric = root.querySelector('#validation-sort-metric');
  const renderRows = (sortKey = 'cost_error', direction = 'asc') => {
    const sorted = [...rows].sort((a, b) => {
      const left = Math.abs(Number(a[sortKey] || 0));
      const right = Math.abs(Number(b[sortKey] || 0));
      return direction === 'asc' ? left - right : right - left;
    });
    tableBody.innerHTML = sorted.map((row) => `<tr><td>${row.project_id || 'Not published'}</td><td>${row.project_name || 'Not reported'}</td><td>${value(row.predicted_cost_overrun)}%</td><td>${value(row.actual_cost_overrun)}%</td><td>${value(row.cost_error)} pp</td><td>${value(row.predicted_delay_days)} days</td><td>${value(row.actual_delay_days)} days</td><td>${value(row.delay_error)} days</td></tr>`).join('');
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

import { api } from '../services/api.js';
import { horizontalBars, lineChart } from '../components/charts.js';

const value = (number, digits = 2) => Number(number || 0).toFixed(digits);

function errorBuckets(rows, key) {
  const buckets = [[0, 5], [5, 15], [15, 30], [30, Infinity]];
  return buckets.map(([from, to]) => ({ label: `${from}–${to === Infinity ? '∞' : to}`, value: rows.filter((row) => { const error = Math.abs(Number(row[key])); return error >= from && error < to; }).length }));
}

export async function PredictionAccuracyPage(root) {
  const [report, validation] = await Promise.all([api.validationReport(), api.predictionValidation(100)]);
  const rows = validation.items;
  const averageConfidence = rows.reduce((total, row) => total + Number(row.model_confidence_percentage || 0), 0) / Math.max(rows.length, 1);
  root.innerHTML = `<header class="page-head"><div><span class="kicker">Historical cutoff backtest</span><h1>Prediction Accuracy Dashboard</h1><p>Each project is frozen before completion; the saved model predicts from that date and is then compared with the known final outcome.</p></div></header>
    <div class="notice"><strong>Verification policy:</strong> ${report.metadata.cutoff_policy}. ${report.metadata.future_information_policy}</div>
    <div class="stat-grid"><article class="stat-card blue"><span class="stat-eyebrow">Cost MAE</span><strong class="stat-value">${value(report.cost_model.MAE)} pp</strong><small class="stat-note">R² ${value(report.cost_model.R2, 3)} · accuracy ${value(report.cost_model.accuracy_percentage, 1)}%</small></article><article class="stat-card amber"><span class="stat-eyebrow">Delay MAE</span><strong class="stat-value">${value(report.delay_model.MAE)} days</strong><small class="stat-note">R² ${value(report.delay_model.R2, 3)} · accuracy ${value(report.delay_model.accuracy_percentage, 1)}%</small></article><article class="stat-card green"><span class="stat-eyebrow">Risk classification</span><strong class="stat-value">${value(report.risk_classification.accuracy * 100, 1)}%</strong><small class="stat-note">Precision ${value(report.risk_classification.precision * 100, 1)}% · recall ${value(report.risk_classification.recall * 100, 1)}%</small></article><article class="stat-card"><span class="stat-eyebrow">Model confidence</span><strong class="stat-value">${value(averageConfidence, 1)}%</strong><small class="stat-note">${report.metadata.confidence_definition}</small></article></div>
    <div class="model-grid"><section class="panel"><div class="panel-head"><div><span class="kicker">Cost escalation</span><h2>Predicted vs actual</h2></div></div><h3>AI predicted</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.predicted_cost_overrun })), { suffix: '%' })}<h3>Actual final outcome</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.actual_cost_overrun })), { suffix: '%' })}</section><section class="panel"><div class="panel-head"><div><span class="kicker">Schedule extension</span><h2>Predicted vs actual</h2></div></div><h3>AI predicted</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.predicted_delay_days })), { suffix: ' days' })}<h3>Actual final outcome</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.actual_delay_days })), { suffix: ' days' })}</section></div>
    <div class="model-grid"><section class="panel"><div class="panel-head"><div><span class="kicker">Cost error</span><h2>Error distribution (percentage points)</h2></div></div>${horizontalBars(errorBuckets(rows, 'cost_error'), { format: (count) => `${count} projects` })}</section><section class="panel"><div class="panel-head"><div><span class="kicker">Delay error</span><h2>Error distribution (days)</h2></div></div>${horizontalBars(errorBuckets(rows, 'delay_error'), { format: (count) => `${count} projects` })}</section></div>
    <section class="panel"><div class="panel-head"><div><span class="kicker">${validation.total} completed projects</span><h2>Project-wise forecast accuracy</h2></div><div class="filters compact-filters"><label>Error metric<select id="validation-sort-metric"><option value="cost_error">Cost error</option><option value="delay_error">Delay error</option></select></label><button class="secondary-btn" id="validation-sort-low">Lowest error first</button><button class="secondary-btn" id="validation-sort-high">Highest error first</button></div></div><div class="table-wrap"><table class="data-table"><thead><tr><th>Project ID</th><th>Prediction date</th><th>Predicted cost</th><th>Actual cost</th><th>Cost error</th><th>Predicted delay</th><th>Actual delay</th><th>Delay error</th></tr></thead><tbody id="validation-rows"></tbody></table></div></section>`;
  const tableBody = root.querySelector('#validation-rows');
  const metric = root.querySelector('#validation-sort-metric');
  const renderRows = (sortKey = 'cost_error', direction = 'asc') => {
    const sorted = [...rows].sort((a, b) => {
      const left = Math.abs(Number(a[sortKey] || 0));
      const right = Math.abs(Number(b[sortKey] || 0));
      return direction === 'asc' ? left - right : right - left;
    });
    tableBody.innerHTML = sorted.map((row) => `<tr><td>${row.project_id}</td><td>${row.prediction_date}</td><td>${value(row.predicted_cost_overrun)}%</td><td>${value(row.actual_cost_overrun)}%</td><td>${value(row.cost_error)} pp</td><td>${value(row.predicted_delay_days)} days</td><td>${value(row.actual_delay_days)} days</td><td>${value(row.delay_error)} days</td></tr>`).join('');
  };
  renderRows();
  root.querySelector('#validation-sort-low').addEventListener('click', () => renderRows(metric.value, 'asc'));
  root.querySelector('#validation-sort-high').addEventListener('click', () => renderRows(metric.value, 'desc'));
}

import { api } from '../services/api.js';
import { horizontalBars, lineChart } from '../components/charts.js';

const value = (number, digits = 2) => Number(number || 0).toFixed(digits);
const models = [
  { id: '2001_2015', label: '2001–2015 model' },
  { id: '2015_2021', label: '2015–2021 model' },
];

function errorBuckets(rows, key) {
  const buckets = [[0, 5], [5, 15], [15, 30], [30, Infinity]];
  return buckets.map(([from, to]) => ({
    label: `${from}–${to === Infinity ? '∞' : to}`,
    value: rows.filter((row) => {
      const error = Math.abs(Number(row[key]));
      return error >= from && error < to;
    }).length,
  }));
}

export async function PredictionAccuracyPage(root) {
  let selected = api.getValidationModel();

  async function render() {
    api.setValidationModel(selected);
    const [report, validation] = await Promise.all([
      api.validationReport(selected),
      api.predictionValidation(100, selected),
    ]);

    const rows = validation.items || [];
    const averageConfidence = rows.reduce((total, row) => total + Number(row.model_confidence_percentage || 0), 0) / Math.max(rows.length, 1);

    root.innerHTML = `<header class="page-head"><div><span class="kicker">Historical cutoff backtest</span><h1>Prediction Accuracy Dashboard</h1><p>Compare predictions against completed projects using the selected historical training window.</p></div></header>
    <section class="panel"><div class="panel-head"><div><span class="kicker">Model selection</span><h2>Validation model window</h2></div></div><select id="validation-model" class="data-select">${models.map((m) => `<option value="${m.id}" ${m.id === selected ? 'selected' : ''}>${m.label}</option>`).join('')}</select></section>
    <div class="notice"><strong>Verification policy:</strong> ${report.metadata.cutoff_policy}. ${report.metadata.future_information_policy}</div>
    <div class="stat-grid"><article class="stat-card blue"><span class="stat-eyebrow">Cost MAE</span><strong class="stat-value">${value(report.cost_model.MAE)} pp</strong></article><article class="stat-card amber"><span class="stat-eyebrow">Delay MAE</span><strong class="stat-value">${value(report.delay_model.MAE)} days</strong></article><article class="stat-card green"><span class="stat-eyebrow">Risk accuracy</span><strong class="stat-value">${value(report.risk_classification.accuracy * 100, 1)}%</strong></article><article class="stat-card"><span class="stat-eyebrow">Confidence</span><strong class="stat-value">${value(averageConfidence, 1)}%</strong></article></div>
    <div class="model-grid"><section class="panel"><h2>Cost predicted vs actual</h2>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.predicted_cost_overrun })), { suffix: '%' })}</section><section class="panel"><h2>Delay predicted vs actual</h2>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.predicted_delay_days })), { suffix: ' days' })}</section></div>
    <div class="model-grid"><section class="panel"><h2>Cost error distribution</h2>${horizontalBars(errorBuckets(rows, 'cost_error'), { format: (count) => `${count} projects` })}</section><section class="panel"><h2>Delay error distribution</h2>${horizontalBars(errorBuckets(rows, 'delay_error'), { format: (count) => `${count} projects` })}</section></div>
    <section class="panel"><h2>${validation.total || rows.length} completed projects</h2><div class="table-wrap"><table class="data-table"><thead><tr><th>Project</th><th>Cutoff</th><th>Cost</th><th>Error</th><th>Delay</th><th>Error</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${row.project_id}</td><td>${row.prediction_date}</td><td>${value(row.predicted_cost_overrun)}% / ${value(row.actual_cost_overrun)}%</td><td>${value(row.cost_error)} pp</td><td>${value(row.predicted_delay_days)} / ${value(row.actual_delay_days)} days</td><td>${value(row.delay_error)} days</td></tr>`).join('')}</tbody></table></div></section>`;

    document.getElementById('validation-model').addEventListener('change', (event) => {
      selected = event.target.value;
      render();
    });
  }

  render();
}

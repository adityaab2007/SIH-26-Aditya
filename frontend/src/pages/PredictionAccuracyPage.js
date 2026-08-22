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
    api.setValidationModel(selected);
    const [report, validation] = await Promise.all([
      api.validationReport(selected),
      api.predictionValidation(100, selected),
    ]);
    const rows = validation.items || [];

    root.innerHTML = `<header class="page-head"><div><span class="kicker">Dynamic temporal validation</span><h1>Prediction Accuracy Dashboard</h1><p>Metrics are loaded from the retrained model generated from the selected year range.</p></div></header>
    <section class="panel"><h2>Model year range</h2><input id="validation-model" class="data-input" value="${selected}" placeholder="Example: 2001_2015"/><button id="load-validation" class="button">Load</button></section>
    <div class="stat-grid"><article class="stat-card blue"><span>Cost MAE</span><strong>${value(report.cost_model.MAE)} pp</strong></article><article class="stat-card amber"><span>Delay MAE</span><strong>${value(report.delay_model.MAE)} days</strong></article><article class="stat-card green"><span>Risk accuracy</span><strong>${value(report.risk_classification.accuracy*100,1)}%</strong></article></div>
    <div class="model-grid"><section class="panel"><h2>Cost prediction</h2>${lineChart(rows.map(r=>({label:r.project_id,value:r.predicted_cost_overrun})),{suffix:'%'})}</section><section class="panel"><h2>Delay prediction</h2>${lineChart(rows.map(r=>({label:r.project_id,value:r.predicted_delay_days})),{suffix:' days'})}</section></div>
    <div class="model-grid"><section class="panel">${horizontalBars(errorBuckets(rows,'cost_error'),{format:v=>`${v} projects`})}</section><section class="panel">${horizontalBars(errorBuckets(rows,'delay_error'),{format:v=>`${v} projects`})}</section></div>`;

    document.getElementById('load-validation').onclick = () => {
      selected = document.getElementById('validation-model').value.trim();
      render();
    };
  }
  render();
}

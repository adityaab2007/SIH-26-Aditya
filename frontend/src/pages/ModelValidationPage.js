import { api } from '../services/api.js';
import { statCard } from '../components/StatCard.js';
import { horizontalBars, lineChart } from '../components/charts.js';

const modelName = (value) => (value || 'not trained').replaceAll('_', ' ');
const num = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';

function comparisonTable(title, rows, regression = false) {
  if (!rows?.length) return '';
  return `<section class="panel"><div class="panel-head"><div><span class="kicker">Same evaluation protocol</span><h2>${title}</h2></div></div>
  <div class="table-wrap"><table class="data-table compact-table"><thead><tr><th>Model</th>${regression ? '<th>MAE</th><th>RMSE</th><th>R²</th>' : '<th>Accuracy</th><th>F1</th><th>ROC-AUC</th>'}<th>Validation</th></tr></thead><tbody>
  ${rows.map(r => `<tr class="${r.is_best ? 'best-row' : ''}"><td><strong>${modelName(r.model)}</strong>${r.is_best ? '<span class="best-tag">best</span>' : ''}</td>${regression ? `<td>${num(r.mae)}</td><td>${num(r.rmse)}</td><td>${num(r.r2)}</td>` : `<td>${num(r.accuracy)}</td><td>${num(r.f1)}</td><td>${num(r.roc_auc)}</td>`}<td>${(r.validation_method || 'baseline').replaceAll('_',' ')}</td></tr>`).join('')}
  </tbody></table></div></section>`;
}

function errorBuckets(rows) {
  const buckets = [
    { label: '0–5', min: 0, max: 5, value: 0 },
    { label: '5–10', min: 5, max: 10, value: 0 },
    { label: '10–25', min: 10, max: 25, value: 0 },
    { label: '25–50', min: 25, max: 50, value: 0 },
    { label: '50+', min: 50, max: Infinity, value: 0 },
  ];
  for (const row of rows || []) {
    const e = Number(row.absolute_error);
    const b = buckets.find(x => e >= x.min && e < x.max);
    if (b) b.value += 1;
  }
  return buckets;
}

function sampleValidation(rows, unit) {
  const row = rows?.find(r => Number.isFinite(Number(r.actual)) && Number.isFinite(Number(r.predicted)));
  if (!row) return `<div class="empty-chart">No out-of-sample validation row available.</div>`;
  return `<div class="detail-grid"><div><span>Project</span><strong>${row.project_name || row.project_code}</strong></div><div><span>AI prediction</span><strong>${num(row.predicted)} ${unit}</strong></div><div><span>Hidden actual outcome</span><strong>${num(row.actual)} ${unit}</strong></div><div><span>Absolute error</span><strong>${num(row.absolute_error)} ${unit}</strong></div></div>`;
}

export async function ModelValidationPage(root) {
  const [summary, backtest, comparison] = await Promise.all([
    api.modelValidation(), api.modelBacktest(), api.modelComparison(),
  ]);
  const best = summary.best_models || {};
  const costReg = best.cost_regressor || {};
  const scheduleReg = best.schedule_regressor || {};
  const costCls = best.cost_classifier || {};
  const scheduleCls = best.schedule_classifier || {};
  const costRows = backtest?.tasks?.cost_regressor?.rows || [];
  const scheduleRows = backtest?.tasks?.schedule_regressor?.rows || [];
  const costActual = costRows.slice(0, 24).map((r,i)=>({label:r.project_code || String(i+1),value:Number(r.actual)}));
  const costPred = costRows.slice(0, 24).map((r,i)=>({label:r.project_code || String(i+1),value:Number(r.predicted)}));
  const forward = summary.forward_validation || {};

  root.innerHTML = `<header class="page-head hero-head"><div><span class="kicker">Proof layer · no hidden leakage</span><h1>Model Validation</h1><p>Out-of-sample evaluation, model-family comparison, prediction-vs-actual evidence and the exact rule used to keep future outcomes hidden from the model.</p></div><a class="primary-btn" href="#/models">Open Model Lab <span>→</span></a></header>
  <div class="notice"><strong>Leakage rule:</strong> ${summary.methodology.forecasting_rule}</div>
  <div class="stat-grid">
    ${statCard({eyebrow:'Cost regression MAE',value:num(costReg.mae),note:`${modelName(costReg.model)} · ${(costReg.validation_method||'baseline').replaceAll('_',' ')}`,icon:'₹',tone:'amber'})}
    ${statCard({eyebrow:'Schedule regression MAE',value:`${num(scheduleReg.mae)} days`,note:`${modelName(scheduleReg.model)} · ${(scheduleReg.validation_method||'baseline').replaceAll('_',' ')}`,icon:'◷',tone:'blue'})}
    ${statCard({eyebrow:'Cost classifier ROC-AUC',value:num(costCls.roc_auc,3),note:modelName(costCls.model),icon:'◎',tone:'green'})}
    ${statCard({eyebrow:'Schedule classifier ROC-AUC',value:num(scheduleCls.roc_auc,3),note:modelName(scheduleCls.model),icon:'△'})}
  </div>
  <div class="dashboard-grid">
    <section class="panel span-2"><div class="panel-head"><div><span class="kicker">Backtest evidence</span><h2>Cost escalation · actual outcomes</h2></div></div>${lineChart(costActual,{suffix:'%'})}</section>
    <section class="panel span-2"><div class="panel-head"><div><span class="kicker">Same held-out rows</span><h2>Cost escalation · model predictions</h2></div></div>${lineChart(costPred,{suffix:'%'})}</section>
    <section class="panel"><div class="panel-head"><div><span class="kicker">Absolute error distribution</span><h2>Cost model errors</h2></div></div>${horizontalBars(errorBuckets(costRows),{format:v=>`${v} projects`})}</section>
    <section class="panel"><div class="panel-head"><div><span class="kicker">Historical proof</span><h2>Sample hidden-outcome check</h2></div></div>${sampleValidation(costRows,'% escalation')}</section>
    <section class="panel span-2"><div class="panel-head"><div><span class="kicker">Forward-horizon readiness</span><h2>${forward.available ? 'Temporal forecast archive ready' : 'Expanded archive still required'}</h2></div></div><p>${forward.available ? `Forward labels are available for ${forward.projects} projects at a ${forward.horizon_months}-month horizon.` : (forward.reason || 'No forward archive metadata available.')}</p><div class="detail-grid"><div><span>Official replay snapshots</span><strong>${forward.snapshot_rows ?? '—'}</strong></div><div><span>Projects with forward pairs</span><strong>${forward.projects ?? '—'}</strong></div><div><span>Schedule labels</span><strong>${forward.schedule_label_rows ?? '—'}</strong></div><div><span>Cost labels</span><strong>${forward.cost_label_rows ?? '—'}</strong></div></div></section>
  </div>
  <header class="page-head"><div><span class="kicker">Algorithm selection</span><h1>Model comparison</h1><p>XGBoost, Random Forest and CatBoost are compared on the same held-out protocol. The production registry selects the best model by ROC-AUC for classification and MAE for regression.</p></div></header>
  <div class="model-grid">
    ${comparisonTable('Cost overrun classifiers',comparison.tasks?.cost_classifier,false)}
    ${comparisonTable('Schedule overrun classifiers',comparison.tasks?.schedule_classifier,false)}
    ${comparisonTable('Cost escalation regressors',comparison.tasks?.cost_regressor,true)}
    ${comparisonTable('Schedule extension regressors',comparison.tasks?.schedule_regressor,true)}
  </div>
  <section class="panel methodology-strip"><div><strong>What is genuinely proven today?</strong><p>${summary.metadata?.forecasting_note || ''}</p></div><a href="#/about">Methodology</a></section>`;
}

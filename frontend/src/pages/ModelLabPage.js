import { api } from '../services/api.js';
import { modelMetrics } from '../features/model-lab/ModelMetrics.js';
import { horizontalBars } from '../components/charts.js';

const fixed = (value, digits = 2) => value === null || value === undefined ? 'n/a' : Number(value).toFixed(digits);
const escape = (value = '') => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

function lifecycleComparison(payload) {
  if (!payload.available) return `<section class="panel"><span class="kicker">Official monthly lifecycle pipeline</span><h2>Comparison pending</h2><div class="notice compact">${escape(payload.reason)}</div></section>`;
  return payload.windows.map((window) => {
    const baseline = window.baseline.metrics;
    const lifecycle = window.lifecycle.metrics;
    const stages = window.lifecycle.lifecycle_stages;
    const evolution = window.forecast_evolution_example || [];
    return `<section class="panel"><div class="panel-head"><div><span class="kicker">Out-of-time window ${escape(window.window)}</span><h2>Five-feature baseline vs monthly lifecycle model</h2></div></div>
      <div class="notice compact"><strong>Evidence boundary:</strong> ${escape(window.metadata.data_source)}. ${escape(window.metadata.leakage_policy)}</div>
      <div class="table-wrap"><table class="data-table compact-table"><thead><tr><th>Model</th><th>Cost MAE</th><th>Cost R²</th><th>Delay MAE</th><th>Delay R²</th><th>Risk macro F1</th></tr></thead><tbody>
      <tr><td>Existing 5-feature baseline</td><td>${fixed(baseline.cost.MAE)} pp</td><td>${fixed(baseline.cost.R2, 3)}</td><td>${fixed(baseline.delay.MAE)} days</td><td>${fixed(baseline.delay.R2, 3)}</td><td>${fixed(baseline.risk.macro_f1 * 100, 1)}%</td></tr>
      <tr><td>Official monthly lifecycle</td><td>${fixed(lifecycle.cost.MAE)} pp</td><td>${fixed(lifecycle.cost.R2, 3)}</td><td>${fixed(lifecycle.delay.MAE)} days</td><td>${fixed(lifecycle.delay.R2, 3)}</td><td>${fixed(lifecycle.risk.macro_f1 * 100, 1)}%</td></tr></tbody></table></div>
      <h3>Early-warning performance by lifecycle stage</h3><div class="table-wrap"><table class="data-table compact-table"><thead><tr><th>Stage</th><th>Cost MAE</th><th>Delay MAE</th><th>Risk macro F1</th></tr></thead><tbody>${['early','mid','late','very_late'].map((stage) => { const item=stages[stage]; return `<tr><td>${stage.replace('_',' ')}</td><td>${item.available ? `${fixed(item.cost.MAE)} pp` : 'Unavailable'}</td><td>${item.available ? `${fixed(item.delay.MAE)} days` : 'Unavailable'}</td><td>${item.available ? `${fixed(item.risk.macro_f1*100,1)}%` : 'Unavailable'}</td></tr>`; }).join('')}</tbody></table></div>
      ${evolution.length ? `<h3>Official historical forecast evolution · ${escape(evolution[0].project_name)}</h3><div class="table-wrap"><table class="data-table compact-table"><thead><tr><th>Snapshot</th><th>Predicted / actual cost</th><th>Predicted / actual delay</th><th>Predicted / actual risk</th></tr></thead><tbody>${evolution.map((row) => `<tr><td>${escape(row.snapshot_date)}</td><td>${fixed(row.predicted_cost_overrun)}% / ${fixed(row.actual_cost_overrun)}%</td><td>${fixed(row.predicted_delay_days)} / ${fixed(row.actual_delay_days)} days</td><td>${escape(row.predicted_risk)} / ${escape(row.actual_risk)}</td></tr>`).join('')}</tbody></table></div>` : '<div class="notice compact">No test project has multiple identity-verified snapshots in this window.</div>'}
      <div class="notice compact"><strong>Ablations generated:</strong> revised-cost exclusion, snapshot-only vs trajectory, and with/without agency priors. Open the versioned evaluation report for complete metrics.</div></section>`;
  }).join('');
}

export async function ModelLabPage(root) {
  const [metrics,importance,monthly]=await Promise.all([api.modelMetrics(),api.modelImportance(),api.monthlyLifecycleComparison()]);
  const imp=(importance.cost_model||[]).slice(0,8).map(i=>({label:i.feature,value:i.importance*100}));
  const examples=(metrics.metadata.prediction_vs_actual||[]).map(x=>`<tr><td>${x.project_id}</td><td>${x.month}</td><td>${x.target.replace('_model','')}</td><td>${x.predicted}</td><td>${x.actual}</td><td>${x.absolute_error}</td></tr>`).join('');
  root.innerHTML=`<header class="page-head"><div><span class="kicker">Temporal validation</span><h1>Model Performance</h1><p>Candidate models are selected on a held-out validation cohort and reported on later-starting test projects.</p></div></header>
  <div class="notice"><strong>Dataset scope:</strong> ${metrics.metadata.dataset_kind}. ${metrics.metadata.split_strategy}</div>
  ${lifecycleComparison(monthly)}
  ${modelMetrics(metrics)}
  <section class="panel"><div class="panel-head"><div><span class="kicker">Held-out example</span><h2>Prediction vs actual</h2></div></div><div class="table-wrap"><table class="data-table compact-table"><thead><tr><th>Project</th><th>Snapshot</th><th>Target</th><th>AI predicted</th><th>Actual</th><th>Error</th></tr></thead><tbody>${examples}</tbody></table></div></section>
  <section class="panel"><div class="panel-head"><div><span class="kicker">Selected cost model</span><h2>Global feature importance · ${metrics.cost_model.best_model.replaceAll('_',' ')}</h2></div></div>${horizontalBars(imp,{format:v=>`${v.toFixed(1)}%`})}</section>`;
}

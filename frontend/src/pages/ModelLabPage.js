import { api } from '../services/api.js';
import { modelMetrics } from '../features/model-lab/ModelMetrics.js';
import { horizontalBars } from '../components/charts.js';

export async function ModelLabPage(root) {
  const [metrics,importance]=await Promise.all([api.modelMetrics(),api.modelImportance()]);
  const imp=(importance.cost_model||[]).slice(0,8).map(i=>({label:i.feature,value:i.importance*100}));
  const examples=(metrics.metadata.prediction_vs_actual||[]).map(x=>`<tr><td>${x.project_id}</td><td>${x.month}</td><td>${x.target.replace('_model','')}</td><td>${x.predicted}</td><td>${x.actual}</td><td>${x.absolute_error}</td></tr>`).join('');
  root.innerHTML=`<header class="page-head"><div><span class="kicker">Temporal validation</span><h1>Model Performance</h1><p>Candidate models are selected on a held-out validation cohort and reported on later-starting test projects.</p></div></header>
  <div class="notice"><strong>Dataset scope:</strong> ${metrics.metadata.dataset_kind}. ${metrics.metadata.split_strategy}</div>
  ${modelMetrics(metrics)}
  <section class="panel"><div class="panel-head"><div><span class="kicker">Held-out example</span><h2>Prediction vs actual</h2></div></div><div class="table-wrap"><table class="data-table compact-table"><thead><tr><th>Project</th><th>Snapshot</th><th>Target</th><th>AI predicted</th><th>Actual</th><th>Error</th></tr></thead><tbody>${examples}</tbody></table></div></section>
  <section class="panel"><div class="panel-head"><div><span class="kicker">Selected cost model</span><h2>Global feature importance · ${metrics.cost_model.best_model.replaceAll('_',' ')}</h2></div></div>${horizontalBars(imp,{format:v=>`${v.toFixed(1)}%`})}</section>`;
}

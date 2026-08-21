import { api } from '../services/api.js';
import { modelMetrics } from '../features/model-lab/ModelMetrics.js';
import { horizontalBars } from '../components/charts.js';

export async function ModelLabPage(root) {
  const [metrics,importance]=await Promise.all([api.modelMetrics(),api.modelImportance()]);
  const key=`schedule_classifier:${metrics.schedule_classifier.best_model}`;
  const imp=(importance[key]||[]).slice(0,8).map(i=>({label:i.feature,value:i.importance*100}));
  root.innerHTML=`<header class="page-head"><div><span class="kicker">Reproducible evaluation</span><h1>Model Lab</h1><p>Four model families are trained for every task. Cross-validated metrics decide which artifact the API uses instead of hard-coding a favourite algorithm.</p></div></header>
  <div class="notice"><strong>Dataset scope:</strong> ${metrics.metadata.dataset_kind}. ${metrics.metadata.forecasting_note}</div>
  ${modelMetrics(metrics)}
  <section class="panel"><div class="panel-head"><div><span class="kicker">Best schedule classifier</span><h2>Global feature importance · ${metrics.schedule_classifier.best_model.replaceAll('_',' ')}</h2></div></div>${horizontalBars(imp,{format:v=>`${v.toFixed(1)}%`})}</section>`;
}

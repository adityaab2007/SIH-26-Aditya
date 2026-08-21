function classifierTable(data) {
  const names = Object.keys(data).filter(k => k !== 'best_model');
  return `<div class="model-block"><h3>Classification</h3><div class="table-wrap"><table class="data-table compact-table"><thead><tr><th>Model</th><th>Rows</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>${names.map(n=>{const m=data[n]; return `<tr class="${n===data.best_model?'best-row':''}"><td><strong>${n.replaceAll('_',' ')}</strong>${n===data.best_model?'<span class="best-tag">best</span>':''}</td><td>${m.rows}</td><td>${m.accuracy}</td><td>${m.f1}</td><td>${m.roc_auc}</td></tr>`}).join('')}</tbody></table></div></div>`;
}
function regressorTable(data) {
  const names = Object.keys(data).filter(k => k !== 'best_model');
  return `<div class="model-block"><h3>Regression</h3><div class="table-wrap"><table class="data-table compact-table"><thead><tr><th>Model</th><th>Rows</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead><tbody>${names.map(n=>{const m=data[n]; return `<tr class="${n===data.best_model?'best-row':''}"><td><strong>${n.replaceAll('_',' ')}</strong>${n===data.best_model?'<span class="best-tag">best</span>':''}</td><td>${m.rows}</td><td>${m.mae}</td><td>${m.rmse}</td><td>${m.r2}</td></tr>`}).join('')}</tbody></table></div></div>`;
}
export function modelMetrics(metrics) {
  return `<div class="model-grid"><section class="panel"><div class="panel-head"><div><span class="kicker">Schedule intelligence</span><h2>Schedule models</h2></div></div>${classifierTable(metrics.schedule_classifier)}${regressorTable(metrics.schedule_regressor)}</section>
  <section class="panel"><div class="panel-head"><div><span class="kicker">Cost intelligence</span><h2>Cost models</h2></div></div>${classifierTable(metrics.cost_classifier)}${regressorTable(metrics.cost_regressor)}</section></div>`;
}

function classifierTable(data) {
  const names = Object.keys(data).filter(k => k !== 'best_model');
  return `<div class="model-block"><h3>Classification</h3><div class="table-wrap"><table class="data-table compact-table"><thead><tr><th>Model</th><th>Rows</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>${names.map(n=>{const m=data[n]; return `<tr class="${n===data.best_model?'best-row':''}"><td><strong>${n.replaceAll('_',' ')}</strong>${n===data.best_model?'<span class="best-tag">best</span>':''}</td><td>${m.rows}</td><td>${m.accuracy}</td><td>${m.f1}</td><td>${m.roc_auc}</td></tr>`}).join('')}</tbody></table></div></div>`;
}
function regressorTable(data) {
  const names = Object.keys(data).filter(k => k !== 'best_model');
  return `<div class="model-block"><h3>Regression</h3><div class="table-wrap"><table class="data-table compact-table"><thead><tr><th>Model</th><th>Rows</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead><tbody>${names.map(n=>{const m=data[n]; return `<tr class="${n===data.best_model?'best-row':''}"><td><strong>${n.replaceAll('_',' ')}</strong>${n===data.best_model?'<span class="best-tag">best</span>':''}</td><td>${m.rows}</td><td>${m.mae}</td><td>${m.rmse}</td><td>${m.r2}</td></tr>`}).join('')}</tbody></table></div></div>`;
}
function temporalTable(label, data) {
  const rows=Object.entries(data.candidates).map(([name, item])=>`<tr class="${name===data.best_model?'best-row':''}"><td><strong>${name.replaceAll('_',' ')}</strong>${name===data.best_model?'<span class="best-tag">selected</span>':''}</td><td>${item.validation.MAE}</td><td>${item.validation.RMSE}</td><td>${item.validation.R2}</td><td>${item.test.MAE}</td><td>${item.test.RMSE}</td><td>${item.test.R2}</td></tr>`).join('');
  return `<section class="panel"><div class="panel-head"><div><span class="kicker">${label}</span><h2>${label} model</h2></div></div><div class="table-wrap"><table class="data-table compact-table"><thead><tr><th>Model</th><th>Validation MAE</th><th>Validation RMSE</th><th>Validation R²</th><th>Test MAE</th><th>Test RMSE</th><th>Test R²</th></tr></thead><tbody>${rows}</tbody></table></div></section>`;
}
export function modelMetrics(metrics) { return `<div class="model-grid">${temporalTable('Cost escalation',metrics.cost_model)}${temporalTable('Schedule delay',metrics.delay_model)}</div>`; }

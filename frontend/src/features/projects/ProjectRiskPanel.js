import { probability, daysHuman, pct } from '../../utils/format.js';
import { riskBadge } from '../../components/RiskBadge.js';

function drivers(items) {
  if (!items?.length) return `<p class="muted">Local SHAP explanation unavailable for this artifact.</p>`;
  return `<div class="driver-list">${items.map((d) => `<div class="driver-row"><div class="driver-head"><span>${d.feature}</span><span class="driver-direction ${d.direction.includes('raises')?'up':'down'}">${d.direction}</span></div><div class="driver-track"><span style="width:${Math.max(5,d.share*100)}%"></span></div></div>`).join('')}</div>`;
}

export function projectRiskPanel(pred) {
  return `<section class="risk-panel panel">
    <div class="panel-head"><div><span class="kicker">Model intelligence</span><h2>Project risk profile</h2></div>${riskBadge(pred.priority_level, `${pred.priority_score}/100 · ${pred.priority_level}`)}</div>
    <div class="risk-grid">
      <div class="risk-metric"><span>Schedule risk signal</span><strong>${probability(pred.schedule_risk_probability)}</strong><small>${pred.best_models.schedule_classifier}</small></div>
      <div class="risk-metric"><span>Cost risk signal</span><strong>${probability(pred.cost_risk_probability)}</strong><small>${pred.best_models.cost_classifier}</small></div>
      <div class="risk-metric"><span>Estimated schedule extension</span><strong>${daysHuman(pred.estimated_schedule_extension_days)}</strong><small>${pred.best_models.schedule_regressor}</small></div>
      <div class="risk-metric"><span>Estimated cost escalation</span><strong>${pct(pred.estimated_cost_escalation_pct)}</strong><small>${pred.best_models.cost_regressor}</small></div>
    </div>
    <div class="notice compact"><strong>Model scope:</strong> ${pred.model_scope}. Current artifacts are trained on observed overrun state from the real May 2026 subset; forward validation is shown separately as the archive grows.</div>
    <div class="two-col drivers-grid"><div><h3>Schedule model drivers</h3>${drivers(pred.schedule_drivers)}</div><div><h3>Cost model drivers</h3>${drivers(pred.cost_drivers)}</div></div>
  </section>`;
}

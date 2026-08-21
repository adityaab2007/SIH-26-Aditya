import { api } from '../services/api.js';
import { riskAlerts, bindAlerts } from '../features/alerts/RiskAlerts.js';

export async function AlertsPage(root) {
  const data=await api.portfolioRisk(100);
  const critical=data.items.filter(i=>i.priority_level==='critical').length;
  root.innerHTML=`<header class="page-head"><div><span class="kicker">Early Warning Queue</span><h1>Prioritise attention, not just red flags.</h1><p>The triage score combines model risk signals with project financial exposure; it is a review-priority index, not a legal or causal conclusion.</p></div><div class="hero-number"><strong>${critical}</strong><span>critical</span></div></header><section class="panel">${riskAlerts(data.items,100)}</section>`;
  bindAlerts(root);
}

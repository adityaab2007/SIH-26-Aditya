import { riskBadge } from '../../components/RiskBadge.js';
import { probability } from '../../utils/format.js';

export function riskAlerts(items, max = 8) {
  return `<div class="alert-list">${items.slice(0,max).map((p,i) => `<button class="alert-row" data-project="${p.project_code}">
      <span class="alert-rank">${String(i+1).padStart(2,'0')}</span>
      <span class="alert-body"><strong>${p.project_name}</strong><small>#${p.project_code} · confidence ${p.confidence}</small></span>
      <span class="alert-signals"><small>Schedule ${probability(p.schedule_risk_probability)}</small><small>Cost ${probability(p.cost_risk_probability)}</small></span>
      ${riskBadge(p.priority_level, `${p.priority_score}`)}
    </button>`).join('')}</div>`;
}

export function bindAlerts(root) {
  root.querySelectorAll('.alert-row[data-project]').forEach((b) => b.addEventListener('click', () => location.hash=`#/project/${b.dataset.project}`));
}

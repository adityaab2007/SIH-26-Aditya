import { moneyCr, daysHuman, pct } from '../../utils/format.js';

export function peerBenchmark(data, project) {
  const m = data.medians;
  const item = (label, projectValue, peerValue, fmt) => `<div class="benchmark-item"><span>${label}</span><div><strong>${fmt(projectValue)}</strong><small>Project</small></div><div><strong>${fmt(peerValue)}</strong><small>Peer median</small></div></div>`;
  return `<section class="panel"><div class="panel-head"><div><span class="kicker">Comparative analytics</span><h2>${data.sector} peer benchmark</h2></div><span class="mini-count">${data.peer_count} peers</span></div>
    <div class="benchmark-grid">
      ${item('Original cost', project.original_cost_cr, m.original_cost_cr, moneyCr)}
      ${item('Cost escalation', project.cost_escalation_pct, m.cost_escalation_pct, pct)}
      ${item('Schedule extension', project.schedule_extension_days, m.schedule_extension_days, daysHuman)}
      ${item('Financial progress', project.financial_progress_pct, m.financial_progress_pct, pct)}
      ${item('Physical progress', project.physical_progress_pct, m.physical_progress_pct, pct)}
    </div>
  </section>`;
}

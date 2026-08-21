import { lineChart } from '../../components/charts.js';
import { moneyCr, pct, dateHuman, probability } from '../../utils/format.js';

export function historyReplay(data) {
  const points = data.snapshots.map(s => ({ label: dateHuman(s.snapshot_date), value: s.physical_progress_pct ?? 0 }));
  const rows = data.snapshots.map((s,i) => `<div class="timeline-node ${i===data.snapshots.length-1?'latest':''}">
    <div class="timeline-dot"></div><div class="timeline-date">${dateHuman(s.snapshot_date)}</div>
    <div class="timeline-card"><div><span>Physical progress</span><strong>${pct(s.physical_progress_pct)}</strong></div><div><span>Expenditure</span><strong>${moneyCr(s.expenditure_cr)}</strong></div><div><span>Revised cost</span><strong>${moneyCr(s.revised_cost_cr)}</strong></div><div><span>Target completion</span><strong>${dateHuman(s.revised_completion_date)}</strong></div><div><span>Baseline priority</span><strong>${s.baseline_priority_score}</strong></div></div>
  </div>`).join('');
  return `<div class="history-replay"><section class="panel"><div class="panel-head"><div><span class="kicker">Official longitudinal snapshots</span><h2>${data.project_name}</h2><p class="muted">#${data.project_code}</p></div></div>${lineChart(points,{suffix:'%', emptyText:'No physical progress series'})}<div class="notice compact">${data.note}</div></section><section class="timeline">${rows}</section></div>`;
}

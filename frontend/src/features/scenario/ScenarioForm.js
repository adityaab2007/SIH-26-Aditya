import { probability } from '../../utils/format.js';

export function scenarioResults(data) {
  const b=data.baseline,s=data.scenario;
  const delta=(s.priority_score-b.priority_score).toFixed(1);
  return `<div class="scenario-result"><div class="scenario-score"><span>Baseline</span><strong>${b.priority_score}</strong><small>${b.priority_level}</small></div><div class="scenario-arrow">→</div><div class="scenario-score scenario-after"><span>Scenario</span><strong>${s.priority_score}</strong><small>${s.priority_level}</small></div><div class="scenario-delta ${Number(delta)<=0?'good':'bad'}">${Number(delta)>0?'+':''}${delta} pts</div></div>
  <div class="scenario-signals"><div><span>Schedule signal</span><strong>${probability(b.schedule_risk_probability)} → ${probability(s.schedule_risk_probability)}</strong></div><div><span>Cost signal</span><strong>${probability(b.cost_risk_probability)} → ${probability(s.cost_risk_probability)}</strong></div></div><div class="notice compact">${data.note}</div>`;
}

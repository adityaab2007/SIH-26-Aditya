import { api } from '../services/api.js';
import { projectRiskPanel } from '../features/projects/ProjectRiskPanel.js';
import { peerBenchmark } from '../features/projects/PeerBenchmark.js';
import { moneyCr, dateHuman, pct, daysHuman } from '../utils/format.js';
import { progressRing } from '../components/charts.js';

export async function ProjectDetailPage(root, code) {
  const [project,pred,peers] = await Promise.all([api.project(code),api.prediction(code),api.peers(code)]);
  root.innerHTML=`<header class="page-head detail-head"><div><a class="back-link" href="#/projects">← Projects</a><span class="kicker">#${project.project_code} · ${project.sector}</span><h1>${project.project_name}</h1><p>${project.ministry}</p></div><a class="secondary-btn" href="#/scenario?project=${project.project_code}">Run scenario</a></header>
  <div class="detail-summary panel"><div class="detail-financial"><div><span>Original cost</span><strong>${moneyCr(project.original_cost_cr)}</strong></div><div><span>Revised cost</span><strong>${moneyCr(project.revised_cost_cr)}</strong></div><div><span>Cumulative expenditure</span><strong>${moneyCr(project.expenditure_cr)}</strong></div><div><span>Cost escalation</span><strong>${pct(project.cost_escalation_pct)}</strong></div></div>
  <div class="detail-rings">${progressRing(project.financial_progress_pct,'financial','progress')}${progressRing(project.physical_progress_pct,'physical','progress')}</div></div>
  <div class="project-meta panel"><div><span>Original completion</span><strong>${dateHuman(project.original_end_date)}</strong></div><div><span>Revised completion</span><strong>${dateHuman(project.revised_end_date)}</strong></div><div><span>Observed extension</span><strong>${daysHuman(project.schedule_extension_days)}</strong></div><div><span>Data confidence</span><strong>${pred.confidence}</strong></div></div>
  ${projectRiskPanel(pred)}${peerBenchmark(peers,project)}
  <section class="source-note"><span>Source</span><a href="${project.source_url}" target="_blank" rel="noreferrer">Official PAIMANA public dashboard ↗</a></section>`;
}

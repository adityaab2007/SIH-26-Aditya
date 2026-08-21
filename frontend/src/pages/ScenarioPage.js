import { api } from '../services/api.js';
import { scenarioResults } from '../features/scenario/ScenarioForm.js';
import { moneyCr, pct } from '../utils/format.js';

export async function ScenarioPage(root) {
  const projects=await api.projects({limit:100});
  const params=new URLSearchParams(location.hash.split('?')[1]||'');
  const preset=params.get('project') || projects.items[0]?.project_code;
  root.innerHTML=`<header class="page-head"><div><span class="kicker">Sensitivity analysis</span><h1>Scenario Explorer</h1><p>Change execution evidence and observe how the trained baseline responds. Outputs are sensitivity estimates—not causal promises.</p></div></header>
  <div class="scenario-layout"><section class="panel"><label>Project<select id="scenario-project">${projects.items.map(p=>`<option value="${p.project_code}" ${p.project_code===preset?'selected':''}>${p.project_name}</option>`).join('')}</select></label><div id="scenario-current"></div><label>Physical progress %<input id="scenario-progress" type="number" min="0" max="100" step="1"></label><label>Cumulative expenditure (₹ Cr)<input id="scenario-exp" type="number" min="0" step="1"></label><button id="run-scenario" class="primary-btn full">Recalculate sensitivity</button></section><section class="panel scenario-output"><div id="scenario-output"><div class="empty-state"><strong>Set a scenario</strong><span>Adjust progress or expenditure, then recalculate.</span></div></div></section></div>`;
  const sel=root.querySelector('#scenario-project'), curr=root.querySelector('#scenario-current'), prog=root.querySelector('#scenario-progress'), exp=root.querySelector('#scenario-exp'), out=root.querySelector('#scenario-output');
  const load=async()=>{const p=await api.project(sel.value); curr.innerHTML=`<div class="current-mini"><span>Current</span><strong>${pct(p.physical_progress_pct)} physical · ${moneyCr(p.expenditure_cr)} spent</strong></div>`; prog.value=p.physical_progress_pct??''; exp.value=p.expenditure_cr??'';};
  sel.addEventListener('change',load); await load();
  root.querySelector('#run-scenario').addEventListener('click',async()=>{out.innerHTML='<div class="loading">Running trained models…</div>'; const data=await api.scenario({project_code:sel.value,physical_progress_pct:prog.value===''?null:Number(prog.value),expenditure_cr:exp.value===''?null:Number(exp.value)}); out.innerHTML=scenarioResults(data);});
}

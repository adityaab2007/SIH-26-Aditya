import { api } from '../services/api.js';
import { historyReplay } from '../features/time-machine/HistoryReplay.js';

export async function TimeMachinePage(root) {
  const list=await api.historyList();
  root.innerHTML=`<header class="page-head"><div><span class="kicker">Historical Forecast Replay</span><h1>Project Time Machine</h1><p>Replay official monthly snapshots to see how cost, expenditure, progress and baseline risk evolve without mixing later values into the earlier view.</p></div></header>
  <div class="notice"><strong>Integrity rule:</strong> these are real official snapshots. The current demo replays baseline model scores but does not claim forward-horizon accuracy until more historical months are ingested.</div>
  <section class="panel time-controls"><label>Select project<select id="history-project">${list.items.map(i=>`<option value="${i.project_code}">${i.project_name} · ${i.snapshots} snapshots</option>`).join('')}</select></label></section><div id="history-content"></div>`;
  const select=root.querySelector('#history-project'), content=root.querySelector('#history-content');
  const load=async()=>{content.innerHTML='<div class="loading">Loading official snapshots…</div>'; content.innerHTML=historyReplay(await api.history(select.value));};
  select.addEventListener('change',load); await load();
}

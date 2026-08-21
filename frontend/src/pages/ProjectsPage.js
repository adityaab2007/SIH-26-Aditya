import { api } from '../services/api.js';
import { projectTable, bindProjectRows } from '../features/projects/ProjectTable.js';

export async function ProjectsPage(root) {
  const first = await api.projects();
  root.innerHTML = `<header class="page-head"><div><span class="kicker">Project registry</span><h1>PAIMANA project explorer</h1><p>Search the curated official May 2026 public-project subset by project, code, ministry or sector.</p></div></header>
  <section class="panel"><div class="filters"><label class="search-box"><span>⌕</span><input id="project-search" placeholder="Search project, code or ministry…"></label><select id="sector-filter"><option value="">All sectors</option>${first.sectors.map(s=>`<option>${s}</option>`).join('')}</select><span id="result-count" class="mini-count">${first.items.length} projects</span></div><div id="project-table">${projectTable(first.items)}</div></section>`;
  bindProjectRows(root);
  const search=root.querySelector('#project-search'), sector=root.querySelector('#sector-filter'), table=root.querySelector('#project-table'), count=root.querySelector('#result-count');
  let timer;
  const refresh=()=>{clearTimeout(timer); timer=setTimeout(async()=>{const data=await api.projects({search:search.value,sector:sector.value}); table.innerHTML=projectTable(data.items); count.textContent=`${data.items.length} projects`; bindProjectRows(root);},180)};
  search.addEventListener('input',refresh); sector.addEventListener('change',refresh);
}

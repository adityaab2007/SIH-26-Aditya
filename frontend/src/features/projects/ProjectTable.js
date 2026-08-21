import { moneyCr, dateHuman, pct, escapeHtml } from '../../utils/format.js';

export function projectTable(items) {
  if (!items.length) return `<div class="empty-state"><strong>No projects found</strong><span>Try another search or sector.</span></div>`;
  return `<div class="table-wrap"><table class="data-table"><thead><tr><th>Project</th><th>Sector</th><th>Original cost</th><th>Revised cost</th><th>Expenditure</th><th>Original end</th><th>Progress</th></tr></thead><tbody>
    ${items.map((p) => `<tr class="clickable-row" data-project="${escapeHtml(p.project_code)}">
      <td><div class="project-cell"><strong>${escapeHtml(p.project_name)}</strong><small>#${escapeHtml(p.project_code)} · ${escapeHtml(p.ministry)}</small></div></td>
      <td><span class="sector-pill">${escapeHtml(p.sector)}</span></td>
      <td>${moneyCr(p.original_cost_cr)}</td><td>${moneyCr(p.revised_cost_cr)}</td><td>${moneyCr(p.expenditure_cr)}</td><td>${dateHuman(p.original_end_date)}</td><td>${pct(p.physical_progress_pct)}</td>
    </tr>`).join('')}
  </tbody></table></div>`;
}

export function bindProjectRows(root) {
  root.querySelectorAll('[data-project]').forEach((el) => el.addEventListener('click', () => { location.hash = `#/project/${el.dataset.project}`; }));
}

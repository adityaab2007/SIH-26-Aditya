import { moneyCr, number } from '../../utils/format.js';
import { statCard } from '../../components/StatCard.js';

export function portfolioOverview(s) {
  return `<div class="stat-grid">
    ${statCard({ eyebrow:'Projects analysed', value:number(s.projects), note:`${s.sectors} represented sectors`, icon:'▦' })}
    ${statCard({ eyebrow:'Original portfolio cost', value:moneyCr(s.original_cost_cr, true), note:'Sum of included official project rows', icon:'₹', tone:'blue' })}
    ${statCard({ eyebrow:'Current cost basis', value:moneyCr(s.current_cost_basis_cr, true), note:'Revised where available, otherwise original', icon:'↗', tone:'amber' })}
    ${statCard({ eyebrow:'Cumulative expenditure', value:moneyCr(s.expenditure_cr, true), note:'Reported expenditure in public rows', icon:'◎', tone:'green' })}
  </div>`;
}

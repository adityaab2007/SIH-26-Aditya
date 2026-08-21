import { api } from '../services/api.js';
import { portfolioOverview } from '../features/dashboard/PortfolioOverview.js';
import { riskAlerts, bindAlerts } from '../features/alerts/RiskAlerts.js';
import { horizontalBars } from '../components/charts.js';

export async function DashboardPage(root) {
  const [summary, risk, projects] = await Promise.all([api.portfolioSummary(), api.portfolioRisk(12), api.projects()]);
  const bySector = Object.entries(projects.items.reduce((a,p)=>{a[p.sector]=(a[p.sector]||0)+1; return a;},{})).map(([label,value])=>({label,value})).sort((a,b)=>b.value-a.value);
  const dist = Object.entries(summary.risk_distribution).map(([label,value])=>({label:label[0].toUpperCase()+label.slice(1),value}));
  root.innerHTML = `<header class="page-head hero-head"><div><span class="kicker">AI for Infrastructure Monitoring · SIH26103</span><h1>Predict risk before overruns harden into outcomes.</h1><p>Real PAIMANA project records, trained ML baselines, explainable scoring, comparative analytics and longitudinal replay in one decision-support workspace.</p></div><a class="primary-btn" href="#/projects">Explore projects <span>→</span></a></header>
    ${portfolioOverview(summary)}
    <div class="dashboard-grid"><section class="panel span-2"><div class="panel-head"><div><span class="kicker">Prioritisation</span><h2>Projects requiring attention</h2></div><a href="#/alerts">View all</a></div>${riskAlerts(risk.items,8)}</section>
    <section class="panel"><div class="panel-head"><div><span class="kicker">Portfolio</span><h2>Risk distribution</h2></div></div>${horizontalBars(dist)}</section>
    <section class="panel"><div class="panel-head"><div><span class="kicker">Coverage</span><h2>Projects by sector</h2></div></div>${horizontalBars(bySector)}</section>
    <section class="panel span-2 methodology-strip"><div><strong>Model honesty built into the product.</strong><p>The currently trained models are baseline overrun-intelligence models on a real May 2026 PAIMANA subset. Historical snapshots are real; future-horizon training is explicitly gated on full OCMS/PAIMANA monthly archive ingestion.</p></div><a href="#/about">Read methodology</a></section></div>`;
  bindAlerts(root);
}

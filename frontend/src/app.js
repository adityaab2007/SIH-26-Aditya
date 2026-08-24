import { sidebar } from './components/Sidebar.js';
import { DashboardPage } from './pages/DashboardPage.js';
import { ProjectsPage } from './pages/ProjectsPage.js';
import { ProjectDetailPage } from './pages/ProjectDetailPage.js';
import { TimeMachinePage } from './pages/TimeMachinePage.js';
import { ModelLabPage } from './pages/ModelLabPage.js';
import { AlertsPage } from './pages/AlertsPage.js';
import { ScenarioPage } from './pages/ScenarioPage.js';
import { DataQualityPage } from './pages/DataQualityPage.js';
import { AssistantPage } from './pages/AssistantPage.js';
import { AboutPage } from './pages/AboutPage.js';
import { ForecastPage } from './pages/ForecastPage.js';
import { PredictionAccuracyPage } from './pages/PredictionAccuracyPage.js';
import { ModelSimulationPage } from './pages/ModelSimulationPage.js';

const routes = {
  dashboard: DashboardPage,
  projects: ProjectsPage,
  forecast: ForecastPage,
  'prediction-accuracy': PredictionAccuracyPage,
  'model-simulation': ModelSimulationPage,
  'time-machine': TimeMachinePage,
  models: ModelLabPage,
  alerts: AlertsPage,
  scenario: ScenarioPage,
  'data-quality': DataQualityPage,
  assistant: AssistantPage,
  about: AboutPage,
};

function parseRoute() {
  const raw=(location.hash || '#/dashboard').slice(2).split('?')[0];
  const parts=raw.split('/').filter(Boolean);
  if(parts[0]==='project' && parts[1]) return {active:'projects', page:'project', arg:parts[1]};
  return {active:parts[0]||'dashboard', page:parts[0]||'dashboard'};
}

async function render() {
  const app=document.querySelector('#app');
  const route=parseRoute();
  app.innerHTML=`${sidebar(route.active)}<main class="main-shell"><div class="topbar"><div class="topbar-title"><span class="status-dot"></span>Decision Support Prototype</div><div class="topbar-meta">MoSPI · PAIMANA · 2026</div></div><div id="page" class="page"><div class="loading">Loading InfraSight…</div></div></main>`;
  const page=app.querySelector('#page');
  try {
    if(route.page==='project') await ProjectDetailPage(page,route.arg);
    else await (routes[route.page]||DashboardPage)(page);
  } catch (err) {
    console.error(err);
    page.innerHTML=`<div class="error-state"><strong>Unable to load this view</strong><span>${err.message}</span><div><button class="secondary-btn" id="retry-view">Retry</button> <a href="#/dashboard">Return to dashboard</a></div></div>`;
    page.querySelector('#retry-view')?.addEventListener('click', render);
  }
  window.scrollTo({top:0,behavior:'instant'});
}

window.addEventListener('hashchange',render);
render();

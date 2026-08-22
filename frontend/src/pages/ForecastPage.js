import { api } from '../services/api.js';
import { horizontalBars } from '../components/charts.js';
import { daysHuman, moneyCr, pct } from '../utils/format.js';

const esc = (value = '') => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');

function factors(items) {
  return horizontalBars((items || []).map((item) => ({ label: `${item.feature} (${item.direction})`, value: Math.abs(item.impact) })), { format: (v) => v.toFixed(3) });
}

export async function ForecastPage(root) {
  const listing = await api.projects({ limit: 100 });
  root.innerHTML = `<header class="page-head"><div><span class="kicker">SIH26103 judging demo</span><h1>Project Forecast</h1><p>Select a project snapshot to see its temporal cost, delay, risk, and model-factor forecast.</p></div></header>
  <section class="panel"><div class="filters"><select id="forecast-project">${listing.items.map((project) => `<option value="${esc(project.project_code)}">${esc(project.project_name)} · ${esc(project.project_code)}</option>`).join('')}</select><button class="primary-btn" id="run-forecast">Generate AI forecast</button></div><div id="forecast-result" class="loading">Select a project and generate its forecast.</div></section>`;
  const select = root.querySelector('#forecast-project');
  const output = root.querySelector('#forecast-result');
  const run = async () => {
    output.innerHTML = '<div class="loading">Building temporal forecast…</div>';
    const forecast = await api.forecast(select.value);
    const status = forecast.current_status;
    output.innerHTML = `<div class="panel-head"><div><span class="kicker">${esc(forecast.project_id)} · ${esc(status.snapshot_month)}</span><h2>${esc(forecast.project_name)}</h2></div><span class="risk-badge risk-${forecast.risk_level.toLowerCase()}"><span class="risk-dot"></span>${forecast.risk_level} · ${forecast.risk_score}/100</span></div>
      <div class="detail-summary"><div class="detail-financial"><div><span>Current estimate</span><strong>${moneyCr(status.current_estimated_cost)}</strong></div><div><span>Physical progress</span><strong>${pct(status.physical_progress_percentage)}</strong></div><div><span>Expected cost increase</span><strong>${pct(forecast.predicted_cost_overrun_percentage)}</strong></div><div><span>Expected delay</span><strong>${daysHuman(forecast.predicted_delay_days)}</strong></div></div></div>
      <div class="notice compact"><strong>Snapshot status:</strong> ${pct(status.progress_delay_percentage_points)} behind planned progress. Planned completion: ${status.planned_completion_date}. ${esc(forecast.model_scope)}</div>
      <div class="two-col drivers-grid"><div><h3>Cost forecast factors · ${esc(forecast.best_models.cost)}</h3>${factors(forecast.cost_factors)}</div><div><h3>Delay forecast factors · ${esc(forecast.best_models.delay)}</h3>${factors(forecast.delay_factors)}</div></div>`;
  };
  root.querySelector('#run-forecast').addEventListener('click', () => run().catch((error) => { output.innerHTML = `<div class="error-state">${esc(error.message)}</div>`; }));
  await run();
}

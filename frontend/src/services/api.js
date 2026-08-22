const JSON_HEADERS = { "Content-Type": "application/json" };

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const type = response.headers.get('content-type') || '';
  if (!response.ok || !type.includes('application/json')) {
    let detail = `Request failed (${response.status})`;
    if (type.includes('application/json')) {
      try { detail = (await response.json()).detail || detail; } catch (_) {}
    }
    throw new Error(detail);
  }
  return response.json();
}

export const api = {
  health: () => request('/api/health'),
  portfolioSummary: () => request('/api/portfolio/summary'),
  portfolioRisk: (limit = 20) => request(`/api/portfolio/risk?limit=${limit}`),
  projects: ({ search = '', sector = '', limit = 100 } = {}) => {
    const p = new URLSearchParams({ limit: String(limit) });
    if (search) p.set('search', search);
    if (sector) p.set('sector', sector);
    return request(`/api/projects?${p.toString()}`);
  },
  project: (code) => request(`/api/projects/${encodeURIComponent(code)}`),
  prediction: (code) => request(`/api/projects/${encodeURIComponent(code)}/prediction`),
  forecast: (code) => request(`/api/projects/${encodeURIComponent(code)}/forecast`),
  peers: (code) => request(`/api/projects/${encodeURIComponent(code)}/peers`),
  modelMetrics: () => request('/api/models/metrics'),
  modelImportance: () => request('/api/models/importance'),
  validationReport: (modelVersion = null) => request(`/api/models/validation${modelVersion ? `?model_version=${encodeURIComponent(modelVersion)}` : ''}`),
  predictionValidation: (limit = 100, modelVersion = null) => request(`/api/models/prediction-validation?limit=${limit}${modelVersion ? `&model_version=${encodeURIComponent(modelVersion)}` : ''}`),
  retrainModel: (startYear, endYear) => request('/api/models/retrain', { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ start_year: Number(startYear), end_year: Number(endYear) }) }),
  simulationVersions: () => request('/api/model-simulations'),
  runSimulation: (version) => request(`/api/model-simulations/${encodeURIComponent(version)}/run`, { method: 'POST' }),
  trainCustomSimulation: (startYear, endYear) => request('/api/model-simulations/custom/train', {
    method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ start_year: Number(startYear), end_year: Number(endYear) }),
  }),
  customSimulationProjects: (sessionId, year) => request(`/api/model-simulations/custom/${encodeURIComponent(sessionId)}/projects?year=${encodeURIComponent(year)}`),
  predictCustomSimulation: (sessionId, recordIndex) => request(`/api/model-simulations/custom/${encodeURIComponent(sessionId)}/predict`, {
    method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ record_index: Number(recordIndex) }),
  }),
  revealCustomSimulation: (sessionId, recordIndex) => request(`/api/model-simulations/custom/${encodeURIComponent(sessionId)}/reveal`, {
    method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ record_index: Number(recordIndex) }),
  }),
  historyList: () => request('/api/history'),
  history: (code) => request(`/api/history/${encodeURIComponent(code)}`),
  scenario: (payload) => request('/api/scenario', { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify(payload) }),
  dataQuality: () => request('/api/data-quality'),
  ask: (query) => request('/api/assistant/query', { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ query }) }),
};

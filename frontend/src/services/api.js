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
  peers: (code) => request(`/api/projects/${encodeURIComponent(code)}/peers`),
  modelMetrics: () => request('/api/models/metrics'),
  modelImportance: () => request('/api/models/importance'),
  modelValidation: () => request('/api/models/validation'),
  modelBacktest: () => request('/api/models/backtest'),
  modelComparison: () => request('/api/models/comparison'),
  modelExplanation: (code) => request(`/api/models/explain/${encodeURIComponent(code)}`),
  historyList: () => request('/api/history'),
  history: (code) => request(`/api/history/${encodeURIComponent(code)}`),
  scenario: (payload) => request('/api/scenario', { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify(payload) }),
  dataQuality: () => request('/api/data-quality'),
  ask: (query) => request('/api/assistant/query', { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ query }) }),
};

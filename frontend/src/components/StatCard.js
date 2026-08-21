export function statCard({ eyebrow, value, note = '', tone = 'neutral', icon = '' }) {
  return `<article class="stat-card ${tone}">
    <div class="stat-card-top"><span class="stat-icon">${icon}</span><span class="stat-eyebrow">${eyebrow}</span></div>
    <div class="stat-value">${value}</div>
    <div class="stat-note">${note}</div>
  </article>`;
}

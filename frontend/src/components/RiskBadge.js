export function riskBadge(level = 'low', label = null) {
  const safe = ['critical', 'high', 'medium', 'low'].includes(level) ? level : 'low';
  const text = label || safe[0].toUpperCase() + safe.slice(1);
  return `<span class="risk-badge risk-${safe}"><span class="risk-dot"></span>${text}</span>`;
}

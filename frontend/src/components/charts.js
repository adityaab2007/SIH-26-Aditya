export function horizontalBars(items, { valueKey = 'value', labelKey = 'label', format = (v) => v, maxItems = 8 } = {}) {
  const rows = items.slice(0, maxItems);
  const max = Math.max(...rows.map((x) => Number(x[valueKey]) || 0), 1);
  return `<div class="hbar-chart">${rows.map((x) => {
    const v = Number(x[valueKey]) || 0;
    return `<div class="hbar-row">
      <div class="hbar-label" title="${x[labelKey]}">${x[labelKey]}</div>
      <div class="hbar-track"><div class="hbar-fill" style="width:${Math.max(2, (v/max)*100)}%"></div></div>
      <div class="hbar-value">${format(v)}</div>
    </div>`;
  }).join('')}</div>`;
}

export function lineChart(points, { width = 760, height = 220, valueKey = 'value', labelKey = 'label', suffix = '', emptyText = 'No trend data' } = {}) {
  if (!points?.length) return `<div class="empty-chart">${emptyText}</div>`;
  const vals = points.map((p) => Number(p[valueKey])).filter(Number.isFinite);
  if (!vals.length) return `<div class="empty-chart">${emptyText}</div>`;
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = Math.max(max - min, 1);
  const padX = 30, padY = 24;
  const x = (i) => padX + (i * (width - padX*2)) / Math.max(points.length - 1, 1);
  const y = (v) => height - padY - ((v - min) / range) * (height - padY*2);
  const path = points.map((p, i) => `${i ? 'L' : 'M'} ${x(i).toFixed(1)} ${y(Number(p[valueKey])).toFixed(1)}`).join(' ');
  const circles = points.map((p, i) => `<circle cx="${x(i)}" cy="${y(Number(p[valueKey]))}" r="4"><title>${p[labelKey]}: ${p[valueKey]}${suffix}</title></circle>`).join('');
  return `<div class="line-chart-wrap"><svg class="line-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Trend chart">
    <line x1="${padX}" y1="${height-padY}" x2="${width-padX}" y2="${height-padY}" class="chart-axis"/>
    <path d="${path}" class="chart-line" fill="none"/>
    <g class="chart-points">${circles}</g>
    <text x="${padX}" y="18" class="chart-minmax">max ${max.toFixed(1)}${suffix}</text>
    <text x="${width-padX}" y="18" text-anchor="end" class="chart-minmax">min ${min.toFixed(1)}${suffix}</text>
  </svg></div>`;
}

export function progressRing(value, label, sublabel = '') {
  const v = Math.max(0, Math.min(100, Number(value) || 0));
  return `<div class="progress-ring" style="--progress:${v * 3.6}deg">
    <div class="progress-ring-inner"><strong>${v.toFixed(0)}</strong><span>${label}</span><small>${sublabel}</small></div>
  </div>`;
}

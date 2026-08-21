export function moneyCr(value, compact = false) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  const n = Number(value);
  if (compact && Math.abs(n) >= 100000) return `₹${(n / 100000).toFixed(2)}L Cr`;
  if (compact && Math.abs(n) >= 1000) return `₹${(n / 1000).toFixed(1)}K Cr`;
  return `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 1 })} Cr`;
}
export function pct(value, digits = 1) { if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'; return `${Number(value).toFixed(digits)}%`; }
export function probability(value) { if (value === null || value === undefined) return '—'; return `${(Number(value) * 100).toFixed(0)}%`; }
export function daysHuman(value) { if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'; const d=Number(value), sign=d<0?'earlier by ':'', a=Math.abs(d); if(a>365)return `${sign}${(a/365.25).toFixed(1)} years`; if(a>60)return `${sign}${Math.round(a/30.44)} months`; return `${sign}${Math.round(a)} days`; }
export function dateHuman(value) { if(!value)return '—'; const d=new Date(`${value}T00:00:00`); if(Number.isNaN(d.getTime()))return value; return d.toLocaleDateString('en-IN',{day:'2-digit',month:'short',year:'numeric'}); }
export function number(value,digits=0){ if(value===null||value===undefined||Number.isNaN(Number(value)))return '—'; return Number(value).toLocaleString('en-IN',{maximumFractionDigits:digits}); }
export function escapeHtml(value=''){ return String(value).replace(/[&<>'"]/g,(c)=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }

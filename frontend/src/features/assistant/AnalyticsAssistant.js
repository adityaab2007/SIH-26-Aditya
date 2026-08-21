export function assistantResponse(data) {
  const rows = (data.items||[]).map(item => `<div class="assistant-item">${Object.entries(item).map(([k,v])=>`<span><small>${k.replaceAll('_',' ')}</small><strong>${v}</strong></span>`).join('')}</div>`).join('');
  return `<div class="assistant-answer"><div class="assistant-avatar">✦</div><div><p>${data.answer}</p>${rows}</div></div>`;
}

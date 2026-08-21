import { api } from '../services/api.js';
import { assistantResponse } from '../features/assistant/AnalyticsAssistant.js';

export async function AssistantPage(root) {
  root.innerHTML=`<header class="page-head"><div><span class="kicker">Analytics interface</span><h1>Project Intelligence Assistant</h1><p>A grounded local assistant over computed project analytics. No external LLM or invented government facts are required for this demo.</p></div></header>
  <section class="assistant-shell panel"><div class="quick-prompts">${['Show highest risk projects','Largest cost overruns','Largest schedule delays','Compare sectors'].map(q=>`<button data-prompt="${q}">${q}</button>`).join('')}</div><div id="assistant-thread"><div class="assistant-welcome"><div>✦</div><p>Ask a portfolio analytics question.</p></div></div><div class="assistant-input"><input id="assistant-query" placeholder="Which projects have the highest risk?"><button id="assistant-send">Send</button></div></section>`;
  const input=root.querySelector('#assistant-query'), thread=root.querySelector('#assistant-thread');
  const send=async(q=input.value)=>{if(!q.trim())return; thread.innerHTML+=`<div class="user-message">${q}</div><div class="loading small">Analysing project data…</div>`; input.value=''; const d=await api.ask(q); thread.querySelector('.loading:last-child')?.remove(); thread.innerHTML+=assistantResponse(d); thread.scrollTop=thread.scrollHeight;};
  root.querySelector('#assistant-send').addEventListener('click',()=>send()); input.addEventListener('keydown',e=>{if(e.key==='Enter')send();}); root.querySelectorAll('[data-prompt]').forEach(b=>b.addEventListener('click',()=>send(b.dataset.prompt)));
}

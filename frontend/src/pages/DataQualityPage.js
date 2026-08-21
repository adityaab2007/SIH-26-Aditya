import { api } from '../services/api.js';
import { dataQualityPanel } from '../features/data-quality/DataQualityPanel.js';

export async function DataQualityPage(root) {
  const data=await api.dataQuality();
  root.innerHTML=`<header class="page-head"><div><span class="kicker">Trust before prediction</span><h1>Data Quality Observatory</h1><p>Government operational data has missing and contradictory fields. InfraSight surfaces those conditions rather than silently overwriting them.</p></div></header>${dataQualityPanel(data)}`;
}

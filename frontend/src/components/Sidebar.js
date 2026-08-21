const NAV = [
  ['dashboard','Dashboard','⌂'],
  ['projects','Projects','▦'],
  ['time-machine','Time Machine','◷'],
  ['models','Model Lab','⌁'],
  ['alerts','Early Warnings','△'],
  ['scenario','Scenario Explorer','⇄'],
  ['data-quality','Data Quality','✓'],
  ['assistant','Intelligence Assistant','✦'],
  ['about','Methodology','i'],
];

export function sidebar(active) {
  return `<aside class="sidebar">
    <div class="brand">
      <div class="brand-mark"><span></span><span></span><span></span></div>
      <div><strong>InfraSight</strong><small>AI · SIH26103</small></div>
    </div>
    <nav class="nav-list">${NAV.map(([route, label, icon]) => `<a class="nav-item ${active===route?'active':''}" href="#/${route}"><span class="nav-icon">${icon}</span><span>${label}</span></a>`).join('')}</nav>
    <div class="sidebar-foot">
      <span class="live-dot"></span>
      <div><strong>Real PAIMANA data</strong><small>May 2026 curated subset</small></div>
    </div>
  </aside>`;
}

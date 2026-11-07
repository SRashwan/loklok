/* LOKLOK thesis site: shared topbar navigation (single source of truth). */
(function () {
  'use strict';
  var nav = document.getElementById('topnav');
  if (!nav) return;
  var base = nav.getAttribute('data-base') || '';

  var primary = [
    { id: 'index', label: 'Overview', href: 'index.html' },
    { id: 'abstract', label: 'Abstract', href: 'abstract.html' },
    { id: 'research', label: 'Research', href: 'research.html' },
    { id: 'results', label: 'Results', href: 'results.html' },
    { id: 'interactive', label: 'Interactive', href: 'interactive/' },
    { id: 'read', label: 'Read Online', href: 'read/' },
  ];
  var more = [
    { id: 'guide', label: 'Reading Guide', href: 'guide.html' },
    { id: 'framework', label: 'Framework', href: 'framework.html' },
    { id: 'pipeline', label: 'The Pipeline', href: 'pipeline.html' },
    { id: 'ratings', label: 'Student Ratings', href: 'ratings.html' },
    { id: 'algorithms', label: 'Algorithms', href: 'algorithms.html' },
    { id: 'blueprint', label: 'Blueprint', href: 'blueprint.html' },
    { id: 'quality', label: 'Quality', href: 'quality.html' },
    { id: 'edition', label: 'Interactive Edition', href: 'edition.html' },
    { id: 'jats', label: 'JATS Reader', href: 'jats.html' },
    { id: 'colab', label: 'Colab Scripts', href: 'colab.html' },
    { id: 'references', label: 'References', href: 'references.html' },
    { id: 'downloads', label: 'Downloads', href: 'downloads.html' },
    { id: 'university', label: 'University &amp; People', href: 'university.html' },
    { id: 'outreach', label: 'Share &amp; Cite', href: 'outreach.html' },
  ];

  function currentId() {
    var p = location.pathname.replace(/\/+$/, '');
    var parts = p.split('/').filter(Boolean);
    var last = parts[parts.length - 1] || 'index';
    if (last === 'thesis') return 'index';
    if (last === 'read' || (parts[parts.length - 2] === 'read' && last === 'index')) return 'read';
    if (parts[parts.length - 1] === 'interactive' || parts[parts.length - 2] === 'interactive') return 'interactive';
    return last.replace(/\.html$/, '');
  }
  var cur = currentId();

  function link(item, cls) {
    var a = document.createElement('a');
    a.href = base + item.href;
    a.textContent = item.label;
    if (cls) a.className = cls;
    if (item.id === cur) a.classList.add('cur');
    return a;
  }

  primary.forEach(function (it) { nav.appendChild(link(it)); });

  var wrap = document.createElement('span');
  wrap.className = 'morewrap';
  var btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'morebtn' + (more.some(function (m) { return m.id === cur; }) ? ' cur' : '');
  btn.setAttribute('aria-expanded', 'false');
  btn.setAttribute('aria-haspopup', 'true');
  btn.textContent = 'More \u25be';
  var menu = document.createElement('div');
  menu.className = 'moremenu';
  menu.setAttribute('role', 'menu');
  more.forEach(function (it) {
    var a = link(it, 'mitem');
    a.setAttribute('role', 'menuitem');
    if (it.id === cur) a.classList.add('cur');
    menu.appendChild(a);
  });
  wrap.appendChild(btn);
  wrap.appendChild(menu);
  nav.appendChild(wrap);

  function close() {
    btn.setAttribute('aria-expanded', 'false');
    menu.classList.remove('open');
    document.removeEventListener('click', close);
  }
  btn.addEventListener('click', function (e) {
    e.stopPropagation();
    var open = menu.classList.toggle('open');
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) setTimeout(function () { document.addEventListener('click', close); }, 0);
  });
  btn.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') { close(); btn.focus(); }
  });
})();

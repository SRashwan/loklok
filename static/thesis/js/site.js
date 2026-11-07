/* LOKLOK thesis site: shared behaviour (classic, subtle, accessible). */
(function () {
  'use strict';

  /* reveal on scroll */
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var els = document.querySelectorAll('.rv');
  if (reduce || !('IntersectionObserver' in window)) {
    els.forEach(function (el) { el.classList.add('in'); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
      });
    }, { threshold: 0.12 });
    els.forEach(function (el) { io.observe(el); });
  }

  /* animated counters */
  var counters = document.querySelectorAll('[data-count]');
  counters.forEach(function (el) {
    var target = parseFloat(el.getAttribute('data-count'));
    var decimals = parseInt(el.getAttribute('data-decimals') || '0', 10);
    var suffix = el.getAttribute('data-suffix') || '';
    if (reduce) { el.textContent = target.toFixed(decimals) + suffix; return; }
    var t0 = null, dur = 900;
    function step(t) {
      if (!t0) t0 = t;
      var p = Math.min(1, (t - t0) / dur);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = (target * eased).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(step);
    }
    var obs = new IntersectionObserver(function (es) {
      es.forEach(function (en) {
        if (en.isIntersecting) { requestAnimationFrame(step); obs.disconnect(); }
      });
    });
    obs.observe(el);
  });

  /* current page marker in top nav */
  var here = location.pathname.replace(/\/+$/, '').split('/').pop() || 'index.html';
  document.querySelectorAll('.topnav a').forEach(function (a) {
    var href = a.getAttribute('href') || '';
    if (href.indexOf(here) === 0 && here !== '') a.classList.add('cur');
  });

  /* thesis data helper: resolve the data file relative to this script,
     so it works from /thesis/ and from /thesis/interactive/ alike */
  var DATA_BASE = (function () {
    var s = document.currentScript ? document.currentScript.src : '';
    var m = s.match(/^(.*\/)js\/site\.js/);
    return m ? m[1] : 'data/';
  })();
  window.THESIS = null;
  window.getThesis = function () {
    if (window.THESIS) return Promise.resolve(window.THESIS);
    var url = DATA_BASE + 'data/thesis_data.json';
    return fetch(url).then(function (r) { return r.json(); })
      .then(function (d) { window.THESIS = d; return d; });
  };
  window.fmt = function (x, d) {
    d = (d === undefined) ? 2 : d;
    if (x === null || x === undefined) return 'n/a';
    return Number(x).toFixed(d);
  };
})();

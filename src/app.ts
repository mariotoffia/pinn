// The hub application.
//
// A hash-routed single page over the pre-rendered chapters that build.mjs inlines into
// window.__PINN__. No framework, no runtime markdown parser, no network access at all - the
// generated index.html works from a file:// URL on a plane.

import type { Bundle, Chapter } from './types';
import { doneCount, getTheme, isDone, recallScroll, rememberScroll, resetProgress, setDone, setTheme, setTourStep, tourStep } from './store';
import { highlight, search } from './search';
import { tourFor } from './tours';
import { ANIMS } from './touranim';

const BUNDLE: Bundle = (window as any).__PINN__;
const CHAPTERS = BUNDLE.chapters;

const $ = (sel: string) => document.querySelector(sel);
const el = (tag: string, cls?: string, html?: string) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html !== undefined) node.innerHTML = html;
  return node;
};
const escapeHtml = (s: string) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

let current: Chapter = CHAPTERS[0];
let disposeAnim = null;
let inTour = false;

function killAnim() {
  if (disposeAnim) {
    try { disposeAnim(); } catch { /* animation already gone */ }
    disposeAnim = null;
  }
}

// ---------------------------------------------------------------------------- theme

function applyTheme() {
  const t = getTheme();
  const root = document.documentElement;
  if (t === 'auto') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', t);
  const btn = $('#theme');
  if (btn) btn.textContent = t === 'auto' ? 'auto' : t;
}

function cycleTheme() {
  const order = ['auto', 'light', 'dark'];
  setTheme(order[(order.indexOf(getTheme()) + 1) % order.length]);
  applyTheme();
}

// ---------------------------------------------------------------------------- sidebar

function renderSidebar() {
  const nav = $('#chapters');
  if (!nav) return;
  nav.innerHTML = '';
  for (const ch of CHAPTERS) {
    const a = el('a', 'chapter' + (ch.slug === current.slug ? ' active' : '') + (isDone(ch.slug) ? ' done' : '')) as HTMLAnchorElement;
    a.href = '#/' + ch.slug;
    a.innerHTML =
      '<span class="tick" aria-hidden="true"></span>' +
      '<span class="ct"><span class="cnum">' + escapeHtml(ch.file.slice(0, 2)) + '</span>' +
      escapeHtml(ch.title) + '</span>' +
      '<span class="cmin">' + ch.minutes + 'm</span>';
    nav.appendChild(a);
  }
  updateProgress();
}

function updateProgress() {
  const n = doneCount();
  const pct = Math.round((n / CHAPTERS.length) * 100);
  const bar = $('#bar');
  const label = $('#progress-label');
  if (bar) (bar as HTMLElement).style.width = pct + '%';
  if (label) label.textContent = n + ' / ' + CHAPTERS.length + ' chapters';
}

// ---------------------------------------------------------------------------- content

function renderTOC(ch: Chapter) {
  const toc = $('#toc');
  if (!toc) return;
  if (!ch.headings.length) {
    toc.innerHTML = '';
    return;
  }
  let html = '<div class="toc-title">On this page</div>';
  for (const h of ch.headings) {
    html += '<a class="h' + h.level + '" href="#' + h.id + '">' + escapeHtml(h.text) + '</a>';
  }
  toc.innerHTML = html;
}

function renderChapter(ch: Chapter) {
  killAnim();
  inTour = false;
  current = ch;
  const main = $('#content');
  if (!main) return;

  const idx = CHAPTERS.indexOf(ch);
  const prev = CHAPTERS[idx - 1];
  const next = CHAPTERS[idx + 1];

  const tour = tourFor(ch.slug);
  const tourLink = tour
    ? '<a class="tourlaunch" href="#/tour/' + ch.slug + '">▶ Guided tour: ' + escapeHtml(tour.title) +
      ' <span>' + tour.minutes + ' min · interactive</span></a>'
    : '';

  const head =
    '<div class="chapter-head">' +
    '<div class="eyebrow">' + escapeHtml(ch.file.replace(/\.md$/, '')) +
    ' · ' + ch.minutes + ' min read · ' + ch.words.toLocaleString() + ' words · ' +
    ch.links + ' links</div>' +
    '<p class="subtitle">' + escapeHtml(ch.subtitle) + '</p>' + tourLink + '</div>';

  const footer =
    '<div class="chapter-foot">' +
    '<button id="mark" class="mark" type="button">' +
    (isDone(ch.slug) ? 'Marked as done ✓' : 'Mark chapter as done') + '</button>' +
    '<div class="pager">' +
    (prev ? '<a href="#/' + prev.slug + '">← ' + escapeHtml(prev.title) + '</a>' : '<span></span>') +
    (next ? '<a href="#/' + next.slug + '">' + escapeHtml(next.title) + ' →</a>' : '<span></span>') +
    '</div></div>';

  main.innerHTML = head + '<article class="prose">' + ch.html + '</article>' + footer;

  if (ch.index) addTableFilter(main);
  wireCopyButtons(main);

  const mark = $('#mark');
  if (mark) {
    mark.addEventListener('click', () => {
      const now = !isDone(ch.slug);
      setDone(ch.slug, now);
      mark.textContent = now ? 'Marked as done ✓' : 'Mark chapter as done';
      renderSidebar();
    });
  }

  renderTOC(ch);
  renderSidebar();
  document.title = ch.title + ' · PINN Learning Path';
}

// ---------------------------------------------------------------------------- tours

function notebookBlock(nb) {
  const badge = nb.kind === 'marimo' ? 'LOCAL LAB · MARIMO' : nb.kind === 'colab' ? 'COLAB / HOSTED' : 'STARTER SCRIPT';
  const label = nb.url
    ? '<a href="' + nb.url + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(nb.label) + '</a>'
    : '<code>' + escapeHtml(nb.label) + '</code>';
  return (
    '<div class="tnotebook">' +
    '<div class="tnb-head"><span class="tnb-badge ' + nb.kind + '">' + badge + '</span>' + label + '</div>' +
    '<div class="tnb-run"><code>' + escapeHtml(nb.run) + '</code></div>' +
    '<div class="tnb-grid">' +
    '<div><b>What it does</b><p>' + nb.what + '</p></div>' +
    '<div><b>What you need first</b><p>' + nb.foundations + '</p></div>' +
    '<div><b>What success looks like</b><p>' + nb.expect + '</p></div>' +
    '</div></div>'
  );
}

function renderTour(slug: string, wantStep: number) {
  const tour = tourFor(slug);
  const ch = CHAPTERS.find((c) => c.slug === slug);
  const main = $('#content');
  if (!tour || !ch || !main) { renderChapter(ch || CHAPTERS[0]); return; }

  killAnim();
  inTour = true;
  current = ch;
  const n = tour.steps.length;
  let idx = wantStep >= 0 ? wantStep : tourStep(slug);
  idx = Math.max(0, Math.min(n - 1, idx));
  setTourStep(slug, idx);
  const step = tour.steps[idx];

  let dots = '';
  for (let i = 0; i < n; i++) {
    dots += '<a class="tdot' + (i === idx ? ' on' : i < idx ? ' seen' : '') + '" href="#/tour/' + slug + '/' + i + '" title="step ' + (i + 1) + '"></a>';
  }

  let nbHtml = '';
  if (step.notebook) nbHtml = notebookBlock(step.notebook);

  const quiz = step.quiz
    ? '<div class="tquiz" id="tquiz"><div class="tq-q">' + step.quiz.q + '</div>' +
      step.quiz.options.map((o, i) => '<button class="tq-opt" data-i="' + i + '">' + o + '</button>').join('') +
      '<div class="tq-explain" id="tq-explain"></div>' +
      '<button class="tq-reveal" id="tq-reveal">show answer</button></div>'
    : '';

  main.innerHTML =
    '<div class="chapter-head"><div class="eyebrow">' + escapeHtml(ch.file.replace(/\.md$/, '')) +
    ' · guided tour · step ' + (idx + 1) + ' / ' + n + '</div>' +
    '<h1 class="ttitle">' + escapeHtml(tour.title) + '</h1>' +
    '<div class="tdots">' + dots + '</div></div>' +
    '<article class="prose tstep">' +
    '<h2 class="tstep-title">' + escapeHtml(step.title) + '</h2>' +
    (step.body || '') +
    '<div id="tanim-host"></div>' +
    quiz +
    nbHtml +
    '</article>' +
    '<div class="tnav">' +
    '<a class="tback" href="#/' + slug + '">↩ back to chapter</a>' +
    '<div class="tnav-btns">' +
    (idx > 0 ? '<a class="tprev" href="#/tour/' + slug + '/' + (idx - 1) + '">◂ previous</a>' : '<span></span>') +
    (idx < n - 1
      ? '<a class="tnext" href="#/tour/' + slug + '/' + (idx + 1) + '">next ▸</a>'
      : '<a class="tnext done" href="#/' + slug + '">finish ✓</a>') +
    '</div></div>';

  if (step.anim && ANIMS[step.anim]) {
    const hostEl = $('#tanim-host');
    if (hostEl) disposeAnim = ANIMS[step.anim](hostEl as HTMLElement);
  }

  if (step.quiz) {
    const q = step.quiz;
    const explain = $('#tq-explain');
    const opts = Array.from(main.querySelectorAll('.tq-opt'));
    const settle = (pick: number) => {
      for (const o of opts) {
        const i = Number((o as HTMLElement).getAttribute('data-i'));
        o.classList.remove('right', 'wrong');
        if (i === q.correct) o.classList.add('right');
        else if (i === pick) o.classList.add('wrong');
      }
      if (explain) {
        explain.innerHTML = (pick === q.correct ? '<b>Right.</b> ' : pick >= 0 ? '<b>Not quite.</b> ' : '') + q.explain;
        (explain as HTMLElement).style.display = 'block';
      }
    };
    for (const o of opts) o.addEventListener('click', () => settle(Number((o as HTMLElement).getAttribute('data-i'))));
    const rev = $('#tq-reveal');
    if (rev) rev.addEventListener('click', () => settle(-1));
  }

  // step list in the toc pane
  const toc = $('#toc');
  if (toc) {
    let html = '<div class="toc-title">Tour steps</div>';
    for (let i = 0; i < n; i++) {
      html += '<a class="h2' + (i === idx ? ' ton' : '') + '" href="#/tour/' + slug + '/' + i + '">' +
        (i + 1) + '. ' + escapeHtml(tour.steps[i].title) + '</a>';
    }
    toc.innerHTML = html;
  }

  renderSidebar();
  window.scrollTo({ top: 0, behavior: 'auto' });
  document.title = tour.title + ' · ' + ch.title + ' · PINN Learning Path';
}

/** The resource index is one long set of tables; give it a live filter. */
function addTableFilter(main: Element) {
  const wrap = el('div', 'filter');
  wrap.innerHTML =
    '<input id="tf" type="search" placeholder="Filter every resource — try: jax, dead, benchmark, colab" ' +
    'autocomplete="off" spellcheck="false"><span id="tf-count"></span>';
  const article = main.querySelector('.prose');
  if (!article) return;
  article.insertBefore(wrap, article.firstChild);

  const input = wrap.querySelector('#tf') as HTMLInputElement;
  const count = wrap.querySelector('#tf-count');
  const rows = Array.from(main.querySelectorAll('tbody tr'));

  const apply = () => {
    const q = input.value.trim().toLowerCase();
    let shown = 0;
    for (const row of rows) {
      const hit = !q || (row.textContent || '').toLowerCase().includes(q);
      (row as HTMLElement).style.display = hit ? '' : 'none';
      if (hit) shown++;
    }
    for (const table of Array.from(main.querySelectorAll('.table-wrap'))) {
      const visible = table.querySelectorAll('tbody tr:not([style*="none"])').length;
      let node = table.previousElementSibling;
      while (node && node.tagName !== 'H2') node = node.previousElementSibling;
      (table as HTMLElement).style.display = visible ? '' : 'none';
      if (node) (node as HTMLElement).style.display = visible ? '' : 'none';
    }
    if (count) count.textContent = shown + ' / ' + rows.length + ' resources';
  };

  input.addEventListener('input', apply);
  apply();
}

function wireCopyButtons(root: Element) {
  for (const btn of Array.from(root.querySelectorAll('[data-copy]'))) {
    btn.addEventListener('click', () => {
      const block = btn.parentElement && btn.parentElement.querySelector('code');
      if (!block) return;
      const text = block.textContent || '';
      const done = () => {
        btn.textContent = 'copied';
        window.setTimeout(() => { btn.textContent = 'copy'; }, 1200);
      };
      try {
        if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, fallback);
        else fallback();
      } catch {
        fallback();
      }
      function fallback() {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done(); } catch { /* nothing more to try */ }
        document.body.removeChild(ta);
      }
    });
  }
}

// ---------------------------------------------------------------------------- search

let searchOpen = false;

function openSearch() {
  searchOpen = true;
  const box = $('#search-panel');
  if (box) box.classList.add('open');
  const input = $('#q') as HTMLInputElement;
  if (input) { input.focus(); input.select(); }
}

function closeSearch() {
  searchOpen = false;
  const box = $('#search-panel');
  if (box) box.classList.remove('open');
}

function runSearch(q: string) {
  const results = $('#results');
  if (!results) return;
  if (!q.trim()) {
    results.innerHTML =
      '<div class="hint">Search every chapter. Try <b>float64</b>, <b>L-BFGS</b>, ' +
      '<b>causal</b>, <b>DeepXDE</b>, <b>weak baselines</b>.</div>';
    return;
  }
  const hits = search(CHAPTERS, q);
  if (!hits.length) {
    results.innerHTML = '<div class="hint">No matches for “' + escapeHtml(q) + '”.</div>';
    return;
  }
  results.innerHTML = hits
    .map(
      (h) =>
        '<a class="hit" href="#/' + h.chapter.slug + '">' +
        '<div class="hit-title">' + escapeHtml(h.chapter.title) + '</div>' +
        '<div class="hit-snip">' + highlight(escapeHtml(h.snippet), q) + '</div></a>'
    )
    .join('');
}

// ---------------------------------------------------------------------------- router

function chapterFor(hash: string) {
  const slug = hash.replace(/^#\/?/, '').split('#')[0];
  return CHAPTERS.find((c) => c.slug === slug) || null;
}

function route() {
  const hash = window.location.hash;
  if (hash && !hash.startsWith('#/')) {
    // an in-page anchor: let the browser handle it
    const target = document.getElementById(hash.slice(1));
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return;
  }
  if (hash.startsWith('#/tour/')) {
    const parts = hash.slice(2).split('/');           // ['tour', slug, step?]
    if (!inTour) rememberScroll(current.slug, window.scrollY);
    let stepArg = -1;
    if (parts.length > 2) stepArg = parseInt(parts[2], 10);
    renderTour(parts[1] || '', stepArg);
    closeSearch();
    return;
  }
  const ch = chapterFor(hash) || CHAPTERS[0];
  if (ch.slug !== current.slug || inTour || !$('#content').innerHTML) {
    if (!inTour) rememberScroll(current.slug, window.scrollY);
    renderChapter(ch);
    const y = recallScroll(ch.slug);
    window.scrollTo({ top: hash.includes('#') && hash.indexOf('#', 2) > 0 ? 0 : y, behavior: 'auto' });
  }
  closeSearch();
}

function keyboard(e: KeyboardEvent) {
  const tag = (e.target as HTMLElement).tagName;
  const typing = tag === 'INPUT' || tag === 'TEXTAREA';

  if (e.key === 'Escape') { closeSearch(); return; }
  if (typing) return;

  if (e.key === '/') { e.preventDefault(); openSearch(); return; }
  if (inTour) {
    const tour = tourFor(current.slug);
    if (tour) {
      const at = tourStep(current.slug);
      if (e.key === '[' || e.key === 'ArrowLeft') {
        if (at > 0) window.location.hash = '#/tour/' + current.slug + '/' + (at - 1);
        return;
      }
      if (e.key === ']' || e.key === 'ArrowRight') {
        if (at < tour.steps.length - 1) window.location.hash = '#/tour/' + current.slug + '/' + (at + 1);
        return;
      }
    }
  }
  const idx = CHAPTERS.indexOf(current);
  if (e.key === '[' && idx > 0) window.location.hash = '#/' + CHAPTERS[idx - 1].slug;
  if (e.key === ']' && idx < CHAPTERS.length - 1) window.location.hash = '#/' + CHAPTERS[idx + 1].slug;
  if (e.key === 't') cycleTheme();
}

// ---------------------------------------------------------------------------- boot

function boot() {
  applyTheme();

  const themeBtn = $('#theme');
  if (themeBtn) themeBtn.addEventListener('click', cycleTheme);

  const openBtn = $('#open-search');
  if (openBtn) openBtn.addEventListener('click', openSearch);

  const q = $('#q') as HTMLInputElement;
  if (q) q.addEventListener('input', () => runSearch(q.value));

  const backdrop = $('#search-panel');
  if (backdrop) {
    backdrop.addEventListener('click', (ev) => {
      if (ev.target === backdrop) closeSearch();
    });
  }

  const reset = $('#reset');
  if (reset) {
    reset.addEventListener('click', () => {
      resetProgress();
      renderSidebar();
      renderChapter(current);
    });
  }

  const burger = $('#burger');
  if (burger) {
    burger.addEventListener('click', () => document.body.classList.toggle('nav-open'));
  }

  const meta = $('#buildinfo');
  if (meta) {
    meta.textContent =
      BUNDLE.build.chapters + ' chapters · ' + BUNDLE.build.words.toLocaleString() + ' words · ' +
      BUNDLE.build.links + ' links · built ' + BUNDLE.build.generated;
  }

  window.addEventListener('hashchange', route);
  document.addEventListener('keydown', keyboard);
  window.addEventListener('beforeunload', () => rememberScroll(current.slug, window.scrollY));

  runSearch('');
  route();
  if (!window.location.hash) renderChapter(CHAPTERS[0]);
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();

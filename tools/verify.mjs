#!/usr/bin/env node
/**
 * verify.mjs - assert that a built index.html is actually usable.
 *
 * The build already fails loudly on invalid JavaScript (`node --check`). This goes further and
 * checks the things a broken *content* pipeline would silently produce: missing chapters,
 * unrendered markdown, empty tours, a dead search index. CI runs it on every push; run it
 * yourself with `node tools/verify.mjs index.html`.
 */

import { readFile } from 'node:fs/promises';

const file = process.argv[2] || 'index.html';
const html = await readFile(file, 'utf8');

const problems = [];
const ok = [];

function check(label, condition, detail) {
  if (condition) ok.push(label);
  else problems.push(label + (detail ? ' -> ' + detail : ''));
}

// ---------------------------------------------------------------- the shell
check('doctype', /^<!doctype html>/i.test(html.trim()));
check('title', html.includes('<title>PINN Learning Path</title>'));
check('self-contained (no external src)', !/<script[^>]+src=/i.test(html));
check('self-contained (no external stylesheet)', !/<link[^>]+rel="stylesheet"/i.test(html));

// ---------------------------------------------------------------- the data
const m = html.match(/<script id="pinn-data" type="application\/json">([\s\S]*?)<\/script>/);
check('embedded chapter bundle', Boolean(m));

if (m) {
  const bundle = JSON.parse(m[1].replace(/<\\\/script/g, '</script').replace(/<\\!--/g, '<!--'));
  const chapters = bundle.chapters || [];

  check('>= 18 chapters', chapters.length >= 18, chapters.length + ' found');
  check('build metadata', Boolean(bundle.build && bundle.build.generated && bundle.build.words > 0));

  for (const c of chapters) {
    if (!c.title) problems.push('chapter ' + c.slug + ' has no title');
    if (!c.html || c.html.length < 200) problems.push('chapter ' + c.slug + ' rendered empty');
    if (!c.text || c.text.length < 200) problems.push('chapter ' + c.slug + ' has no search text');
    if (/\]\(http/.test(c.html)) problems.push('chapter ' + c.slug + ' has an unrendered markdown link');
    if (/^\s*\|\s*-{3,}/m.test(c.html)) problems.push('chapter ' + c.slug + ' has an unrendered table');
    if (/\$\$/.test(c.html)) problems.push('chapter ' + c.slug + ' has unrendered display math');
  }

  // every internal cross-reference must resolve to a real chapter
  const slugs = new Set(chapters.map((c) => c.slug));
  for (const c of chapters) {
    for (const ref of c.html.matchAll(/href="#\/([a-z0-9-]+)"/g)) {
      if (!slugs.has(ref[1]) && ref[1] !== 'tour') {
        problems.push('chapter ' + c.slug + ' links to missing chapter #/' + ref[1]);
      }
    }
  }
  check('internal cross-references resolve', !problems.some((p) => p.includes('links to missing')));
}

// ---------------------------------------------------------------- the app
check('tour engine present', html.includes('function renderTour'));
check('tour data present', html.includes("slug: '08-pinn-core'"));
check('animation registry present', html.includes('backprop-graph') && html.includes('pinn-live'));
check('measured tour data present', html.includes('BETA_DATA') || html.includes('vanilla_final'));
check('search present', html.includes('function runSearch'));

// ---------------------------------------------------------------- report
const size = (Buffer.byteLength(html) / 1024).toFixed(0);
if (problems.length) {
  console.error('\nFAILED ' + file + ' (' + size + ' KB)\n');
  for (const p of problems) console.error('  x ' + p);
  console.error('');
  process.exit(1);
}
console.log('verified ' + file + ' (' + size + ' KB): ' + ok.length + ' checks passed');

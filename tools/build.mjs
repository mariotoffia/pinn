#!/usr/bin/env node
/**
 * build.mjs - generate a single self-contained index.html from every markdown file in
 * content/ and every TypeScript/JavaScript file in src/.
 *
 * Design constraints, in order of importance:
 *   1. ZERO dependencies. `node tools/build.mjs` works on a machine with nothing installed.
 *      esbuild is used if it happens to be resolvable, purely as a faster/stricter path.
 *   2. The output is ONE file that works from a file:// URL, offline, with no CDN.
 *   3. Markdown is rendered at build time, so the page ships HTML rather than a parser.
 *   4. The build validates its own output with `node --check`, so a type-stripping bug is a
 *      loud failure instead of a blank page.
 *
 * Usage:  node tools/build.mjs [--out dist/index.html] [--quiet] [--date YYYY-MM-DD]
 */

import { readdir, readFile, writeFile, mkdtemp, mkdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { frontMatter, renderMarkdown, toPlainText } from './markdown.mjs';
import { stripTypes } from './striptypes.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const CONTENT = path.join(ROOT, 'content');
const SRC = path.join(ROOT, 'src');

/** Module order for the concatenated bundle. Dependencies first. */
const MODULE_ORDER = ['types.ts', 'store.ts', 'search.ts', 'tourdata.ts', 'tours.ts', 'touranim.ts', 'app.ts'];

const args = process.argv.slice(2);
const outFile = argValue('--out') || 'dist/index.html';
const quiet = args.includes('--quiet');

function argValue(flag) {
  const i = args.indexOf(flag);
  return i !== -1 && args[i + 1] ? args[i + 1] : null;
}

/**
 * The build stamps a date into the page footer. To keep `make generate` byte-reproducible
 * (CI builds twice and diffs), that date can be pinned:
 *   --date 2026-08-24        explicit flag, wins over everything
 *   SOURCE_DATE_EPOCH=...    the cross-ecosystem reproducible-builds convention
 * Otherwise it is simply today.
 */
function buildDate() {
  const flag = argValue('--date');
  if (flag) return flag;
  const epoch = process.env.SOURCE_DATE_EPOCH;
  if (epoch && /^\d+$/.test(epoch)) {
    return new Date(Number(epoch) * 1000).toISOString().slice(0, 10);
  }
  return new Date().toISOString().slice(0, 10);
}

function log(...parts) {
  if (!quiet) console.log(...parts);
}

const escapeScript = (s) => s.replace(/<\/script/gi, '<\\/script').replace(/<!--/g, '<\\!--');

// --------------------------------------------------------------------------- content

async function loadChapters() {
  const files = (await readdir(CONTENT)).filter((f) => f.endsWith('.md')).sort();
  if (!files.length) throw new Error(`no markdown files in ${CONTENT}`);

  const chapters = [];
  let markdownBytes = 0;

  for (const file of files) {
    const raw = await readFile(path.join(CONTENT, file), 'utf8');
    markdownBytes += Buffer.byteLength(raw);
    const [meta, body] = frontMatter(raw);
    const { html, headings } = renderMarkdown(body);
    const text = toPlainText(body);
    const links = (raw.match(/https?:\/\/[^\s)>\]]+/g) || []).length;

    chapters.push({
      slug: file.replace(/\.md$/, ''),
      file,
      title: meta.title || file,
      subtitle: meta.subtitle || '',
      minutes: Number(meta.minutes || Math.max(1, Math.round(text.split(/\s+/).length / 220))),
      index: meta.index === 'true',
      html,
      headings,
      text,
      words: text.split(/\s+/).filter(Boolean).length,
      links,
    });
  }
  return { chapters, markdownBytes };
}

// --------------------------------------------------------------------------- script

function toModuleSource(src) {
  let out = stripTypes(src);
  // The bundle is a plain concatenation inside one IIFE, so module syntax has to go.
  out = out.replace(/^[ \t]*import[\s\S]*?from\s+['"][^'"]+['"];[ \t]*\r?\n?/gm, '');
  out = out.replace(/^[ \t]*import\s+['"][^'"]+['"];[ \t]*\r?\n?/gm, '');
  out = out.replace(/^export\s+(?=(function|const|let|var|class|async|default))/gm, '');
  return out;
}

function esbuildAvailable() {
  try {
    execFileSync('npx', ['--no-install', 'esbuild', '--version'], { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

async function buildScript() {
  const present = new Set(await readdir(SRC));
  const ordered = MODULE_ORDER.filter((f) => present.has(f));
  const extras = [...present]
    .filter((f) => (f.endsWith('.ts') || f.endsWith('.js')) && !MODULE_ORDER.includes(f))
    .sort();
  const files = [...ordered, ...extras];
  if (!files.length) throw new Error(`no .ts/.js sources in ${SRC}`);

  const useEsbuild = !args.includes('--no-esbuild') && esbuildAvailable();
  let body;

  if (useEsbuild) {
    // Let esbuild do the transform, one file at a time, then concatenate as above. We do not
    // use its bundler so that both paths produce byte-comparable structure.
    const dir = await mkdtemp(path.join(tmpdir(), 'pinnbuild-'));
    try {
      const parts = [];
      for (const f of files) {
        const src = await readFile(path.join(SRC, f), 'utf8');
        const tmp = path.join(dir, f);
        await writeFile(tmp, src);
        const js = execFileSync(
          'npx',
          ['--no-install', 'esbuild', tmp, '--loader:.ts=ts', '--format=esm', '--target=es2020'],
          { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }
        );
        parts.push(
          '/* ' + f + ' */\n' +
            js
              .replace(/^[ \t]*import[\s\S]*?from\s+["'][^"']+["'];[ \t]*\r?\n?/gm, '')
              .replace(/^export\s+/gm, '')
        );
      }
      body = parts.join('\n');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  } else {
    const parts = [];
    for (const f of files) {
      const src = await readFile(path.join(SRC, f), 'utf8');
      parts.push('/* ' + f + ' */\n' + toModuleSource(src));
    }
    body = parts.join('\n');
  }

  return {
    js: '(function () {\n"use strict";\n' + body + '\n})();',
    files,
    tool: useEsbuild ? 'esbuild' : 'tools/striptypes.mjs',
  };
}

/** Fail loudly rather than shipping a broken page. */
function validate(js) {
  try {
    execFileSync(process.execPath, ['--check', '-'], { input: js, stdio: 'pipe' });
  } catch (err) {
    const msg = (err.stderr || Buffer.from('')).toString();
    throw new Error(
      'generated JavaScript is not valid:\n' + msg +
        '\nIf this came from the built-in type stripper, install esbuild ' +
        '(`npm i -D esbuild`) and re-run `make generate`.'
    );
  }
}

// --------------------------------------------------------------------------- page

function page({ bundleJson, css, js, build }) {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PINN Learning Path</title>
<meta name="description" content="From neural networks to physics-informed neural networks: a verified, engineer-focused learning path.">
<meta name="generator" content="tools/build.mjs">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ctext y='26' font-size='26'%3E%E2%88%82%3C/text%3E%3C/svg%3E">
<style>
${css}
</style>
</head>
<body>
<header>
  <button id="burger" type="button" aria-label="Toggle chapters">☰</button>
  <div class="brand"><b>PINN Learning Path</b><span>neural networks → physics-informed neural networks</span></div>
  <div class="spacer"></div>
  <button class="searchbtn" id="open-search" type="button">Search<kbd>/</kbd></button>
  <button id="theme" type="button" title="Cycle theme (t)">auto</button>
</header>

<div class="layout">
  <aside class="nav">
    <div class="progress">
      <div class="track"><div id="bar"></div></div>
      <div class="row"><span id="progress-label">0 / 0 chapters</span><button id="reset" type="button">reset</button></div>
    </div>
    <nav id="chapters"></nav>
  </aside>

  <main id="content"></main>

  <aside class="toc" id="toc"></aside>
</div>

<footer class="meta"><span id="buildinfo"></span> · generated by <code>make generate</code> from <code>content/*.md</code> + <code>src/*.ts</code></footer>

<div id="search-panel">
  <div class="search-box">
    <input id="q" type="search" placeholder="Search the whole path…" autocomplete="off" spellcheck="false">
    <div id="results"></div>
  </div>
</div>

<script id="pinn-data" type="application/json">${escapeScript(bundleJson)}</script>
<script>
window.__PINN__ = JSON.parse(document.getElementById('pinn-data').textContent);
</script>
<script>
${js}
</script>
</body>
</html>
`;
}

// --------------------------------------------------------------------------- main

async function main() {
  const t0 = Date.now();
  const { chapters, markdownBytes } = await loadChapters();
  const { js, files, tool } = await buildScript();
  validate(js);

  const css = existsSync(path.join(SRC, 'styles.css'))
    ? await readFile(path.join(SRC, 'styles.css'), 'utf8')
    : '';

  const build = {
    generated: buildDate(),
    chapters: chapters.length,
    words: chapters.reduce((n, c) => n + c.words, 0),
    links: chapters.reduce((n, c) => n + c.links, 0),
    markdownBytes,
    tool,
  };

  const bundleJson = JSON.stringify({ chapters, build });
  const html = page({ bundleJson, css, js, build });
  const out = path.resolve(ROOT, outFile);
  await mkdir(path.dirname(out), { recursive: true });   // so --out dist/... just works
  await writeFile(out, html);

  log(`  markdown : ${chapters.length} chapters, ${(markdownBytes / 1024).toFixed(0)} KB`);
  log(`  scripts  : ${files.join(', ')}  (via ${tool})`);
  log(`  words    : ${build.words.toLocaleString()}   links: ${build.links}`);
  log(`  output   : ${path.relative(process.cwd(), out)}  (${(Buffer.byteLength(html) / 1024).toFixed(0)} KB)`);
  log(`  done in  : ${Date.now() - t0} ms`);
}

main().catch((err) => {
  console.error('\nbuild failed: ' + err.message + '\n');
  process.exit(1);
});

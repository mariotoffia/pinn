/**
 * A small, deliberate Markdown -> HTML renderer.
 *
 * It handles exactly what this curriculum uses: front matter, ATX headings with slug ids,
 * fenced code, tables, blockquotes, nested lists, horizontal rules, `$$ display math $$`,
 * and inline emphasis / code / links. It is not CommonMark-complete and does not try to be -
 * the corpus is authored alongside it, which is what makes a 250-line renderer defensible.
 *
 * Rendering happens at BUILD time, so the published page ships plain HTML: no runtime parser,
 * no CDN, and the markdown files stay the single source of truth.
 */

import { renderDisplayMath } from './latex.mjs';

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const CODE_MARK = '\uE000';
const LINK_MARK = '\uE001';

export function slug(text) {
  return text
    .toLowerCase()
    .trim()
    .replace(/[`*_[\]()]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

export function frontMatter(src) {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(src);
  if (!m) return [{}, src];
  const meta = {};
  for (const line of m[1].split(/\r?\n/)) {
    const kv = /^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$/.exec(line);
    if (kv) meta[kv[1]] = kv[2].replace(/^["']|["']$/g, '');
  }
  return [meta, src.slice(m[0].length)];
}

/** Inline: code, bold, italic, links, bare URLs. Order matters. */
export function inline(src) {
  const codes = [];
  let s = src.replace(/`([^`]+)`/g, (_, c) => {
    codes.push('<code>' + esc(c) + '</code>');
    return CODE_MARK + (codes.length - 1) + CODE_MARK;
  });

  s = esc(s);

  const links = [];
  const stash = (html) => {
    links.push(html);
    return LINK_MARK + (links.length - 1) + LINK_MARK;
  };

  // [text](url)
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, text, url) => {
    const ext = /^https?:/i.test(url);
    const attrs = ext ? ' target="_blank" rel="noopener noreferrer"' : '';
    return stash(
      '<a class="' + (ext ? 'ext' : 'int') + '" href="' + url + '"' + attrs + '>' + text + '</a>'
    );
  });

  // bare URLs. Very long ones are SHOWN abbreviated - the href is always complete, so copying
  // the link still gives you the real thing; only the visible text is trimmed so a table cell
  // does not blow out to 300px of query string.
  s = s.replace(/(^|[\s(])(https?:\/\/[^\s<>)\]]+[^\s<>)\].,;:])/g, (_, pre, url) =>
    pre + stash('<a class="ext" href="' + url + '" target="_blank" rel="noopener noreferrer">' +
      shortenUrl(url) + '</a>')
  );

  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, '$1<em>$2</em>');
  s = s.replace(/~~([^~]+)~~/g, '<del>$1</del>');

  s = s.replace(new RegExp(LINK_MARK + '(\\d+)' + LINK_MARK, 'g'), (_, i) => links[+i]);
  s = s.replace(new RegExp(CODE_MARK + '(\\d+)' + CODE_MARK, 'g'), (_, i) => codes[+i]);
  return s;
}

/** Display form for a long bare URL: host + last meaningful path segment. */
function shortenUrl(url) {
  if (url.length <= 62) return url;
  try {
    const u = new URL(url);
    const parts = u.pathname.split('/').filter(Boolean);
    const tail = parts.length ? parts[parts.length - 1] : '';
    const short = u.host.replace(/^www\./, '') + (tail ? '/…/' + tail : '');
    return short.length < url.length ? short : url.slice(0, 60) + '…';
  } catch {
    return url.slice(0, 60) + '…';
  }
}

function renderTable(rows) {
  const cells = (line) => line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
  const head = cells(rows[0]);
  const align = cells(rows[1]).map((c) =>
    c.startsWith(':') && c.endsWith(':') ? 'center' : c.endsWith(':') ? 'right' : 'left'
  );
  const body = rows.slice(2).map(cells);
  const th = head
    .map((c, i) => '<th style="text-align:' + (align[i] || 'left') + '">' + inline(c) + '</th>')
    .join('');
  const tb = body
    .map(
      (r) =>
        '<tr>' +
        r.map((c, i) => '<td style="text-align:' + (align[i] || 'left') + '">' + inline(c) + '</td>').join('') +
        '</tr>'
    )
    .join('');
  return (
    '<div class="table-wrap"><table><thead><tr>' + th + '</tr></thead><tbody>' + tb + '</tbody></table></div>'
  );
}

/** Build a nested <ul>/<ol> from lines already known to be list items. */
function renderList(items) {
  let html = '';
  const stack = [];
  for (const it of items) {
    while (stack.length && it.indent < stack[stack.length - 1].indent) {
      html += stack.pop().ordered ? '</li></ol>' : '</li></ul>';
    }
    if (!stack.length || it.indent > stack[stack.length - 1].indent) {
      html += it.ordered ? '<ol>' : '<ul>';
      stack.push({ indent: it.indent, ordered: it.ordered });
    } else {
      html += '</li>';
    }
    html += '<li>' + inline(it.text);
  }
  while (stack.length) html += stack.pop().ordered ? '</li></ol>' : '</li></ul>';
  return html;
}

export function renderMarkdown(src) {
  const lines = src.replace(/\r\n/g, '\n').split('\n');
  const out = [];
  const headings = [];
  const para = [];
  let i = 0;

  const flushParagraph = () => {
    if (para.length) out.push('<p>' + inline(para.join(' ')) + '</p>');
    para.length = 0;
  };

  while (i < lines.length) {
    const line = lines[i];

    // fenced code
    const fence = /^```(\w*)\s*$/.exec(line);
    if (fence) {
      flushParagraph();
      const lang = fence[1] || 'text';
      const body = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) body.push(lines[i++]);
      i++;
      out.push(
        '<div class="code"><button class="copy" type="button" data-copy>copy</button>' +
          '<span class="lang">' + esc(lang) + '</span><pre><code>' + esc(body.join('\n')) + '</code></pre></div>'
      );
      continue;
    }

    // display math
    if (line.trim().startsWith('$$')) {
      flushParagraph();
      const t = line.trim();
      if (t.length > 4 && t.endsWith('$$')) {
        out.push(renderDisplayMath(t.slice(2, -2)));
        i++;
      } else {
        const body = [t.slice(2)];
        i++;
        while (i < lines.length && !lines[i].trim().endsWith('$$')) body.push(lines[i++]);
        if (i < lines.length) body.push(lines[i].trim().replace(/\$\$$/, ''));
        i++;
        out.push(renderDisplayMath(body.join(' ')));
      }
      continue;
    }

    // heading
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      flushParagraph();
      const level = h[1].length;
      const text = h[2].trim();
      const id = slug(text);
      if (level >= 2 && level <= 3) headings.push({ level, text: text.replace(/[*`]/g, ''), id });
      const anchor = level >= 2
        ? '<a class="anchor" href="#' + id + '" aria-hidden="true">#</a>'
        : '';
      out.push('<h' + level + ' id="' + id + '">' + anchor + inline(text) + '</h' + level + '>');
      i++;
      continue;
    }

    // horizontal rule
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      flushParagraph();
      out.push('<hr>');
      i++;
      continue;
    }

    // table
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|?\s*$/.test(lines[i + 1])) {
      flushParagraph();
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      out.push(renderTable(rows));
      continue;
    }

    // blockquote
    if (/^\s*>/.test(line)) {
      flushParagraph();
      const buf = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        buf.push(lines[i].replace(/^\s*>\s?/, ''));
        i++;
      }
      out.push('<blockquote>' + renderMarkdown(buf.join('\n')).html + '</blockquote>');
      continue;
    }

    // list
    if (/^(\s*)([-*+]|\d+[.)])\s+(.*)$/.test(line)) {
      flushParagraph();
      const items = [];
      while (i < lines.length) {
        const m = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/.exec(lines[i]);
        if (m) {
          items.push({
            indent: Math.floor(m[1].length / 2),
            ordered: /\d/.test(m[2]),
            text: m[3],
          });
          i++;
        } else if (items.length && /^\s{2,}\S/.test(lines[i])) {
          items[items.length - 1].text += ' ' + lines[i].trim(); // lazy continuation
          i++;
        } else {
          break;
        }
      }
      out.push(renderList(items));
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      i++;
      continue;
    }

    para.push(line.trim());
    i++;
  }
  flushParagraph();
  return { html: out.join('\n'), headings };
}

/** Strip markdown to plain text - used to build the client-side search index. */
export function toPlainText(src) {
  return src
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/https?:\/\/\S+/g, ' ')
    .replace(/[#>|*_`~-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// Client-side full-text search over the whole curriculum.
//
// The index is built at load time from the plain-text version of each chapter that the build
// script emits. It is a few hundred kilobytes of text - small enough that a linear scan is
// instant and a real inverted index would be premature.

import type { Chapter, SearchHit } from './types';

function terms(query: string): string[] {
  return query
    .toLowerCase()
    .split(/[^a-z0-9+#._-]+/)
    .filter((t) => t.length > 1);
}

function countOccurrences(haystack: string, needle: string): number {
  let n = 0;
  let i = haystack.indexOf(needle);
  while (i !== -1) {
    n++;
    i = haystack.indexOf(needle, i + needle.length);
  }
  return n;
}

function makeSnippet(text: string, term: string): string {
  const i = text.toLowerCase().indexOf(term);
  if (i === -1) return text.slice(0, 160) + '…';
  const from = Math.max(0, i - 70);
  const to = Math.min(text.length, i + term.length + 110);
  return (from > 0 ? '…' : '') + text.slice(from, to).trim() + (to < text.length ? '…' : '');
}

export function search(chapters: Chapter[], query: string): SearchHit[] {
  const ts = terms(query);
  if (!ts.length) return [];

  const hits = [];
  for (const ch of chapters) {
    const hayTitle = (ch.title + ' ' + ch.subtitle).toLowerCase();
    const hayText = ch.text.toLowerCase();
    let score = 0;
    let matchedAll = true;

    for (const t of ts) {
      const inTitle = countOccurrences(hayTitle, t);
      const inText = countOccurrences(hayText, t);
      if (!inTitle && !inText) {
        matchedAll = false;
        break;
      }
      score += inTitle * 25 + Math.min(inText, 30);
    }
    if (!matchedAll) continue;

    hits.push({ chapter: ch, score, snippet: makeSnippet(ch.text, ts[0]) });
  }

  hits.sort((a, b) => b.score - a.score);
  return hits.slice(0, 40);
}

/** Highlight query terms inside an already-escaped snippet. */
export function highlight(snippet: string, query: string): string {
  let out = snippet;
  for (const t of terms(query)) {
    const re = new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig');
    out = out.replace(re, '<mark>$1</mark>');
  }
  return out;
}

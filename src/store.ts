// Per-viewer state: which chapters are marked done, and the theme.
//
// Every read and write is wrapped, because localStorage throws outright in a few contexts
// (private windows with site data blocked, some embedded previews). The page must render
// correctly with no stored value at all - progress is a convenience, never load-bearing.

const KEY = 'pinn-path-v1';

let cache = null;

function load() {
  if (cache) return cache;
  cache = { done: {}, theme: 'auto', pos: {}, tour: {} };
  try {
    const raw = window.localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') {
        cache.done = parsed.done && typeof parsed.done === 'object' ? parsed.done : {};
        cache.theme = typeof parsed.theme === 'string' ? parsed.theme : 'auto';
        cache.pos = parsed.pos && typeof parsed.pos === 'object' ? parsed.pos : {};
        cache.tour = parsed.tour && typeof parsed.tour === 'object' ? parsed.tour : {};
      }
    }
  } catch {
    /* no storage available - carry on with defaults */
  }
  return cache;
}

function persist() {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(cache));
  } catch {
    /* quota, private mode, or blocked site data - not worth telling the user about */
  }
}

export function isDone(slug: string): boolean {
  return Boolean(load().done[slug]);
}

export function setDone(slug: string, value: boolean): void {
  load().done[slug] = value;
  if (!value) delete cache.done[slug];
  persist();
}

export function doneCount(): number {
  return Object.keys(load().done).length;
}

export function resetProgress(): void {
  load().done = {};
  persist();
}

export function getTheme(): string {
  return load().theme;
}

export function setTheme(theme: string): void {
  load().theme = theme;
  persist();
}

export function tourStep(slug: string): number {
  const v = load().tour[slug];
  return typeof v === 'number' ? v : 0;
}

export function setTourStep(slug: string, i: number): void {
  load().tour[slug] = i;
  persist();
}

export function rememberScroll(slug: string, y: number): void {
  load().pos[slug] = Math.round(y);
  persist();
}

export function recallScroll(slug: string): number {
  const y = load().pos[slug];
  return typeof y === 'number' ? y : 0;
}

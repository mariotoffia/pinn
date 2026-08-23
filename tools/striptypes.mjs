/**
 * A small TypeScript -> JavaScript type stripper.
 *
 * WHY THIS EXISTS: `make generate` must work on a machine with nothing but Node - no npm
 * install, no network, no toolchain. esbuild is used when it happens to be available (see
 * build.mjs); this is the zero-dependency fallback.
 *
 * It is a character scanner that understands strings, template literals, comments and regex
 * literals, so it never edits inside them. It handles exactly the TypeScript subset this app
 * uses, and the build then runs `node --check` on the result, so a stripping failure is a loud
 * build error rather than a silently broken page.
 *
 * Handled:
 *   import type { A, B } from '...';       -> removed
 *   interface Foo { ... }                  -> removed
 *   type Foo = ...;                        -> removed
 *   export interface / export type         -> removed
 *   name: Type   (params, declarations)    -> annotation removed
 *   ): Type {    (return types)            -> annotation removed
 *   expr as Type                           -> cast removed
 *   ident!                                 -> non-null assertion removed
 *
 * NOT handled (and deliberately unused in src/): enums, decorators, namespaces, generics on
 * call sites, abstract classes, parameter properties, overload signatures.
 */

const ID_START = /[A-Za-z_$]/;
const ID_CHAR = /[A-Za-z0-9_$]/;

/** Tokens after which a `/` starts a regex literal rather than a division. */
const REGEX_PRECEDERS = new Set([
  '(', ',', '=', ':', '[', '!', '&', '|', '?', '{', '}', ';', '+', '-', '*', '%', '<', '>',
  '~', '^', 'return', 'typeof', 'instanceof', 'in', 'of', 'new', 'delete', 'void', 'case', 'do',
  'else', 'yield', 'await',
]);

class Scanner {
  constructor(src) {
    this.s = src;
    this.i = 0;
    this.out = '';
    this.lastSignificant = '';
  }

  get rest() {
    return this.s.slice(this.i);
  }

  emit(text) {
    this.out += text;
    const t = text.trim();
    if (t) this.lastSignificant = t.slice(-1) === ' ' ? this.lastSignificant : t;
  }

  /** Copy a string, template literal, comment or regex verbatim. Returns true if it consumed one. */
  copyOpaque() {
    const s = this.s;
    const c = s[this.i];

    if (c === '/' && s[this.i + 1] === '/') {
      const end = s.indexOf('\n', this.i);
      const stop = end === -1 ? s.length : end;
      this.out += s.slice(this.i, stop);
      this.i = stop;
      return true;
    }
    if (c === '/' && s[this.i + 1] === '*') {
      const end = s.indexOf('*/', this.i + 2);
      const stop = end === -1 ? s.length : end + 2;
      this.out += s.slice(this.i, stop);
      this.i = stop;
      return true;
    }
    if (c === '"' || c === "'") {
      let j = this.i + 1;
      while (j < s.length && s[j] !== c) j += s[j] === '\\' ? 2 : 1;
      this.out += s.slice(this.i, j + 1);
      this.i = j + 1;
      this.lastSignificant = 'str';
      return true;
    }
    if (c === '`') {
      let j = this.i + 1;
      let depth = 0;
      while (j < s.length) {
        if (s[j] === '\\') { j += 2; continue; }
        if (s[j] === '$' && s[j + 1] === '{') { depth++; j += 2; continue; }
        if (s[j] === '}' && depth > 0) { depth--; j++; continue; }
        if (s[j] === '`' && depth === 0) break;
        j++;
      }
      this.out += s.slice(this.i, j + 1);
      this.i = j + 1;
      this.lastSignificant = 'str';
      return true;
    }
    if (c === '/' && REGEX_PRECEDERS.has(this.lastSignificant)) {
      let j = this.i + 1;
      let inClass = false;
      while (j < s.length) {
        if (s[j] === '\\') { j += 2; continue; }
        if (s[j] === '[') inClass = true;
        else if (s[j] === ']') inClass = false;
        else if (s[j] === '/' && !inClass) break;
        else if (s[j] === '\n') return false;
        j++;
      }
      while (j + 1 < s.length && /[a-z]/.test(s[j + 1])) j++;
      this.out += s.slice(this.i, j + 1);
      this.i = j + 1;
      this.lastSignificant = 'regex';
      return true;
    }
    return false;
  }
}

/** Skip a balanced `{...}` starting at index i. Returns the index just past the closing brace. */
function skipBraces(s, i) {
  let depth = 0;
  for (let j = i; j < s.length; j++) {
    if (s[j] === '"' || s[j] === "'" || s[j] === '`') {
      const q = s[j];
      j++;
      while (j < s.length && s[j] !== q) j += s[j] === '\\' ? 2 : 1;
      continue;
    }
    if (s[j] === '{') depth++;
    else if (s[j] === '}') {
      depth--;
      if (depth === 0) return j + 1;
    }
  }
  return s.length;
}

/**
 * Consume a type expression starting at index i, stopping at a delimiter at nesting depth 0.
 * `stops` is the set of characters that end the annotation in this position.
 */
function skipType(s, i, stops) {
  let depth = 0;
  let j = i;
  while (j < s.length) {
    const c = s[j];

    if (c === '"' || c === "'" || c === '`') {
      const q = c;
      j++;
      while (j < s.length && s[j] !== q) j += s[j] === '\\' ? 2 : 1;
      j++;
      continue;
    }

    // The stop check MUST come before the bracket bookkeeping, or a `{` that ends a return-type
    // annotation gets treated as the start of an object type and the whole function body is
    // swallowed. That failure is silent in the output and loud in `node --check`.
    if (depth === 0 && stops.has(c)) return j;

    if (c === '(' || c === '[' || c === '{' || c === '<') { depth++; j++; continue; }
    if (c === ')' || c === ']' || c === '}' || c === '>') {
      if (depth === 0) return j;                 // we have left the enclosing construct
      depth--;
      j++;
      continue;
    }
    if (depth === 0) {
      if (c === '=' && s[j + 1] === '>') { j += 2; continue; }   // function type, not an end
      if (c === '\n') {
        const m = /^\s*([^\s])/.exec(s.slice(j));
        if (!m || !/[|&.[({A-Za-z_$"\'`]/.test(m[1])) return j;
      }
    }
    j++;
  }
  return j;
}

export function stripTypes(src) {
  const sc = new Scanner(src);
  const s = src;

  while (sc.i < s.length) {
    if (sc.copyOpaque()) continue;

    const rest = sc.rest;

    // import type { ... } from '...';
    let m = /^import\s+type\s[\s\S]*?;/.exec(rest);
    if (m && atStatementStart(sc)) { sc.i += m[0].length; continue; }

    // (export )?interface Name ... { ... }
    m = /^(export\s+)?interface\s+[A-Za-z_$][\w$]*\s*(extends\s[^{]*)?\{/.exec(rest);
    if (m && atStatementStart(sc)) {
      sc.i = skipBraces(s, sc.i + m[0].length - 1);
      continue;
    }

    // (export )?type Name = ... ;
    m = /^(export\s+)?type\s+[A-Za-z_$][\w$]*\s*=/.exec(rest);
    if (m && atStatementStart(sc)) {
      const end = skipType(s, sc.i + m[0].length, new Set([';']));
      sc.i = end + 1;
      continue;
    }

    const c = s[sc.i];

    // `as Type` cast
    m = /^\bas\s+(?!const\b)/.exec(rest);
    if (m && /[\w$)\]'"`]/.test(sc.lastSignificant.slice(-1))) {
      sc.i = skipType(s, sc.i + m[0].length, new Set([',', ')', ';', ']', '}', '\n']));
      continue;
    }
    if (/^\bas\s+const\b/.test(rest)) { sc.i += 8; continue; }

    // non-null assertion:  foo!.bar   /   foo!)
    if (c === '!' && /[\w$)\]]/.test(sc.lastSignificant.slice(-1)) && /^![.)\],;\s]/.test(rest)) {
      sc.i += 1;
      continue;
    }

    // optional parameter / property marker:  name?: Type
    if (c === '?' && /^\?\s*:/.test(rest)) {
      sc.i += 1;
      continue;
    }

    // type annotation after `)` -> return type
    if (c === ':' && sc.lastSignificant.slice(-1) === ')') {
      sc.i = skipType(s, sc.i + 1, new Set(['{', '=', ';', ',', ')']));
      continue;
    }

    // type annotation after an identifier in a declaration position
    if (c === ':' && ID_CHAR.test(sc.lastSignificant.slice(-1)) && inDeclarationPosition(sc)) {
      sc.i = skipType(s, sc.i + 1, new Set([',', ')', ';', '=', '\n']));
      continue;
    }

    sc.emit(c);
    sc.i += 1;
  }
  return sc.out;
}

function atStatementStart(sc) {
  const tail = sc.out.replace(/\s+$/, '');
  return tail === '' || /[;{}]$/.test(tail) || /\n\s*$/.test(sc.out);
}

/**
 * Is this `:` a type annotation rather than an object-literal key or a ternary branch?
 *
 * The app source deliberately avoids the ambiguous cases, so this only has to recognise the
 * two positions it actually uses: inside a parameter list, and after `let`/`const`/`var`.
 */
function inDeclarationPosition(sc) {
  const before = sc.out;
  const line = before.slice(before.lastIndexOf('\n') + 1);

  // `let x: T` / `const x: T` / `var x: T`
  if (/\b(let|const|var)\s+[A-Za-z_$][\w$]*\s*$/.test(line)) return true;

  // Walk back to the enclosing bracket. A `?` seen at depth 0 on the way means this colon is
  // the second half of a ternary, not an annotation - that is the only genuinely ambiguous
  // case in ordinary code, and it is worth handling exactly rather than by heuristic.
  let depth = 0;
  for (let k = before.length - 1; k >= 0; k--) {
    const ch = before[k];
    if (ch === ')' || ch === ']' || ch === '}') { depth++; continue; }
    if (ch === '(' || ch === '[' || ch === '{') {
      if (depth === 0) return ch === '(';        // parameter list -> annotation
      depth--;
      continue;
    }
    if (depth === 0) {
      if (ch === '?') return false;              // ternary
      if (ch === ';') return false;
    }
  }
  return false;
}

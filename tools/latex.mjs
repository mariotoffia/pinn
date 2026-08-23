/**
 * A deliberately small LaTeX -> HTML renderer for display math.
 *
 * Scope: exactly the subset used by this curriculum - greek letters, \partial, \frac, \sum,
 * \int, sub/superscripts, \mathcal, \nabla, norms, and the usual relation/operator symbols.
 * It is NOT a TeX engine. It exists so the generated page stays a single self-contained file
 * with no CDN, no KaTeX bundle and no web fonts, while still reading as mathematics.
 *
 * Anything it does not recognise is passed through escaped, so an unhandled macro shows up as
 * literal text rather than silently disappearing.
 */

const SYMBOLS = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', zeta: 'ζ',
  eta: 'η', theta: 'θ', vartheta: 'ϑ', iota: 'ι', kappa: 'κ', lambda: 'λ', mu: 'μ', nu: 'ν',
  xi: 'ξ', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', upsilon: 'υ', phi: 'φ', varphi: 'φ',
  chi: 'χ', psi: 'ψ', omega: 'ω',
  Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ',
  Upsilon: 'Υ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω',
  partial: '∂', nabla: '∇', infty: '∞', cdot: '·', times: '×', pm: '±', mp: '∓',
  leq: '≤', le: '≤', geq: '≥', ge: '≥', neq: '≠', ne: '≠', approx: '≈', sim: '∼',
  equiv: '≡', propto: '∝', in: '∈', notin: '∉', subset: '⊂', subseteq: '⊆',
  rightarrow: '→', to: '→', leftarrow: '←', Rightarrow: '⇒', mapsto: '↦', longmapsto: '⟼',
  forall: '∀', exists: '∃', emptyset: '∅', cup: '∪', cap: '∩', setminus: '∖',
  sum: '∑', prod: '∏', int: '∫', iint: '∬', oint: '∮', sqrt: '√',
  ldots: '…', cdots: '⋯', dots: '…', quad: ' ', qquad: '  ', star: '★', ast: '∗',
  langle: '⟨', rangle: '⟩', lVert: '‖', rVert: '‖', vert: '|', Vert: '‖',
  left: '', right: '', displaystyle: '', text: '', mathrm: '', operatorname: '', bigl: '',
  bigr: '', Bigl: '', Bigr: '', big: '', Big: '', hat: '', bar: '', tilde: '',
};

const CAL = {
  A: '𝒜', B: 'ℬ', C: '𝒞', D: '𝒟', E: 'ℰ', F: 'ℱ', G: '𝒢', H: 'ℋ', I: 'ℐ', J: '𝒥',
  K: '𝒦', L: 'ℒ', M: 'ℳ', N: '𝒩', O: '𝒪', P: '𝒫', Q: '𝒬', R: 'ℛ', S: '𝒮', T: '𝒯',
  U: '𝒰', V: '𝒱', W: '𝒲', X: '𝒳', Y: '𝒴', Z: '𝒵',
};

const BB = { R: 'ℝ', N: 'ℕ', Z: 'ℤ', Q: 'ℚ', C: 'ℂ', E: '𝔼', P: 'ℙ' };

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const FUNCS = ['exp', 'log', 'ln', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'min', 'max',
  'arg', 'argmin', 'argmax', 'det', 'tr', 'diag', 'lim', 'sup', 'inf', 'dim', 'ker', 'deg'];

/** Read the argument that follows `_` or `^`: a {group}, a \macro, or a single character. */
function readGroup(src, i) {
  if (src[i] === '\\') {
    const m = /^\\([A-Za-z]+)/.exec(src.slice(i));
    if (m) return [m[0], i + m[0].length];
  }
  if (src[i] !== '{') return [src[i], i + 1];
  let depth = 0;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}') {
      depth--;
      if (depth === 0) return [src.slice(i + 1, j), j + 1];
    }
  }
  return [src.slice(i + 1), src.length];
}

export function renderLatex(src) {
  let out = '';
  let i = 0;
  while (i < src.length) {
    const c = src[i];

    if (c === '\\') {
      const m = /^\\([A-Za-z]+)/.exec(src.slice(i));
      if (!m) {                                  // \\, \{, \}, \| ...
        const next = src[i + 1];
        if (next === '\\') { out += '<br>'; i += 2; continue; }
        if (next === '|') { out += '‖'; i += 2; continue; }
        out += esc(next ?? ''); i += 2; continue;
      }
      const name = m[1];
      i += m[0].length;

      if (name === 'frac' || name === 'dfrac' || name === 'tfrac') {
        let a, b;
        [a, i] = readGroup(src, i);
        [b, i] = readGroup(src, i);
        out += `<span class="frac"><span class="num">${renderLatex(a)}</span>` +
               `<span class="den">${renderLatex(b)}</span></span>`;
        continue;
      }
      if (name === 'mathcal' || name === 'mathbb' || name === 'mathbf' || name === 'mathit') {
        let g; [g, i] = readGroup(src, i);
        const table = name === 'mathcal' ? CAL : name === 'mathbb' ? BB : null;
        if (table) out += g.split('').map((ch) => table[ch] ?? esc(ch)).join('');
        else out += `<span class="${name}">${renderLatex(g)}</span>`;
        continue;
      }
      if (name === 'text' || name === 'mathrm' || name === 'operatorname') {
        let g; [g, i] = readGroup(src, i);
        out += `<span class="mtext">${esc(g)}</span>`;
        continue;
      }
      if (name === 'sqrt') {
        let g; [g, i] = readGroup(src, i);
        out += `√<span class="sqrt">${renderLatex(g)}</span>`;
        continue;
      }
      if (name === 'begin' || name === 'end') {                 // aligned/array: drop the wrapper
        let g; [g, i] = readGroup(src, i);
        void g;
        continue;
      }
      if (FUNCS.includes(name)) { out += `<span class="mop">${name}</span>`; continue; }
      if (name in SYMBOLS) { out += SYMBOLS[name]; continue; }
      out += esc('\\' + name);
      continue;
    }

    if (c === '^' || c === '_') {
      let g; [g, i] = readGroup(src, i + 1);
      out += c === '^' ? `<sup>${renderLatex(g)}</sup>` : `<sub>${renderLatex(g)}</sub>`;
      continue;
    }
    if (c === '{' || c === '}') { i++; continue; }
    if (c === '&') { out += '<span class="malign"></span>'; i++; continue; }
    if (c === '|') { out += '|'; i++; continue; }
    out += esc(c);
    i++;
  }
  return out;
}

export function renderDisplayMath(src) {
  const rows = src.split(/\\\\/).map((r) => renderLatex(r.trim())).filter(Boolean);
  const body = rows.length > 1
    ? rows.map((r) => `<div class="mrow">${r}</div>`).join('')
    : (rows[0] ?? '');
  return `<div class="math">${body}</div>`;
}

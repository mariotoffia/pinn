// Interactive animations for the guided tours. Zero dependencies, canvas/SVG only,
// theme-aware (colors are read from CSS variables at draw time). Every factory
// receives a host element and returns a dispose function.
//
// The "live" demos are real: the spectral-bias MLP and the 1D Poisson PINN train
// genuinely in the browser (hand-rolled forward/backward in float64 JS). The
// chapter-09 charts play back measured data baked in tourdata.ts.

import { BETA_DATA } from './tourdata';

// ---------------------------------------------------------------- tiny helpers

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

function mk(tag: string, cls?: string, html?: string): HTMLElement {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}

function btn(label: string, onClick) {
  const b = mk('button', 'tbtn', label);
  b.addEventListener('click', onClick);
  return b;
}

function sliderRow(label: string, min: number, max: number, step: number, value: number) {
  const row = mk('div', 'trow');
  const lab = mk('span', 'tlab', label);
  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(min); input.max = String(max); input.step = String(step); input.value = String(value);
  const out = mk('span', 'tval', String(value));
  row.appendChild(lab); row.appendChild(input); row.appendChild(out);
  return { row, input, out };
}

function makeCanvas(host: HTMLElement, height: number) {
  const c = document.createElement('canvas');
  c.className = 'tcanvas';
  host.appendChild(c);
  const ctx = c.getContext('2d');
  function resize() {
    const w = host.clientWidth || 680;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr); c.height = Math.round(height * dpr);
    c.style.width = w + 'px'; c.style.height = height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);
  return { c, ctx, cleanup: () => window.removeEventListener('resize', resize), width: () => c.clientWidth, height: () => height };
}

// deterministic RNG (mulberry32) + gaussian
function rng(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) {
  let u = 0, v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ------------------------------------------------------------- 1. backprop graph

function animBackpropGraph(host: HTMLElement) {
  const x = 2.0, w = -1.5, b = 1.0, y = -0.5;
  const z = w * x + b;
  const a = Math.tanh(z);
  const L = (a - y) * (a - y);
  const dLda = 2 * (a - y);
  const dadz = 1 - a * a;
  const dLdz = dLda * dadz;
  const dLdw = dLdz * x;
  const dLdb = dLdz;
  const dLdx = dLdz * w;
  const f = (v: number) => (Math.round(v * 1000) / 1000).toString();

  const msgs = [
    'The program as a graph. Nothing computed yet — press step ▸.',
    'Forward 1/3: z = w·x + b = ' + f(z),
    'Forward 2/3: a = tanh(z) = ' + f(a),
    'Forward 3/3: L = (a − y)² = ' + f(L) + '. The forward pass is done.',
    'Backward 1/4: ∂L/∂a = 2(a − y) = ' + f(dLda) + ' — the loss’s own local derivative.',
    'Backward 2/4: ∂L/∂z = ∂L/∂a · (1 − a²) = ' + f(dLdz) + ' — chain rule, one arrow.',
    'Backward 3/4: ∂L/∂w = ∂L/∂z · x = ' + f(dLdw) + '   and   ∂L/∂b = ' + f(dLdb) + '. These update the weights.',
    'Backward 4/4: ∂L/∂x = ∂L/∂z · w = ' + f(dLdx) + ' — the gradient with respect to the INPUT. A PINN is built on this number.',
  ];
  let stage = 0;

  const wrap = mk('div', 'tanim');
  const controls = mk('div', 'tcontrols');
  const msg = mk('div', 'tmsg', msgs[0]);
  const svgBox = mk('div', 'tsvg');
  wrap.appendChild(controls); wrap.appendChild(svgBox); wrap.appendChild(msg);
  host.appendChild(wrap);

  function node(nx: number, ny: number, label: string, val: string, on: boolean, grad: string, gradOn: boolean, hot?: boolean) {
    const g = ['<g>'];
    let stroke = cssVar('--line-strong');
    if (hot && gradOn) stroke = cssVar('--accent');
    let valCol = cssVar('--fg-faint');
    if (on) valCol = cssVar('--ok');
    g.push('<rect x="' + (nx - 34) + '" y="' + (ny - 20) + '" width="68" height="40" rx="9" fill="' + cssVar('--bg-raised') + '" stroke="' + stroke + '" stroke-width="' + (hot && gradOn ? 2 : 1.2) + '"/>');
    g.push('<text x="' + nx + '" y="' + (ny - 5) + '" text-anchor="middle" font-size="12" font-weight="600" fill="' + cssVar('--fg') + '">' + label + '</text>');
    g.push('<text x="' + nx + '" y="' + (ny + 11) + '" text-anchor="middle" font-size="11" font-family="monospace" fill="' + valCol + '">' + (on ? val : '·') + '</text>');
    if (gradOn) g.push('<text x="' + nx + '" y="' + (ny + 34) + '" text-anchor="middle" font-size="10.5" font-family="monospace" fill="' + cssVar('--accent') + '">' + grad + '</text>');
    g.push('</g>');
    return g.join('');
  }
  function arrow(x1: number, y1: number, x2: number, y2: number) {
    return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" stroke="' + cssVar('--line-strong') + '" stroke-width="1.4" marker-end="url(#arr)"/>';
  }
  function draw() {
    const s = stage;
    const svg = [
      '<svg viewBox="0 0 680 210" style="width:100%;height:auto;display:block">',
      '<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="' + cssVar('--line-strong') + '"/></marker></defs>',
      arrow(114, 60, 216, 88), arrow(114, 105, 216, 100), arrow(114, 150, 216, 112),
      arrow(284, 100, 366, 100), arrow(434, 100, 516, 100), arrow(550, 170, 550, 124),
      node(80, 60, 'x = 2', '', true, '∂L/∂x = ' + f(dLdx), s >= 7, true),
      node(80, 105, 'w = −1.5', '', true, '∂L/∂w = ' + f(dLdw), s >= 6),
      node(80, 150, 'b = 1', '', true, '∂L/∂b = ' + f(dLdb), s >= 6),
      node(250, 100, 'z = wx+b', f(z), s >= 1, '∂L/∂z = ' + f(dLdz), s >= 5),
      node(400, 100, 'a = tanh z', f(a), s >= 2, '∂L/∂a = ' + f(dLda), s >= 4),
      node(550, 100, 'L = (a−y)²', f(L), s >= 3, '∂L/∂L = 1', s >= 4),
      node(550, 185, 'y = −0.5', '', true, '', false),
      '</svg>',
    ];
    svgBox.innerHTML = svg.join('');
    msg.textContent = msgs[stage];
  }
  controls.appendChild(btn('◂ back', () => { stage = Math.max(0, stage - 1); draw(); }));
  controls.appendChild(btn('step ▸', () => { stage = Math.min(msgs.length - 1, stage + 1); draw(); }));
  controls.appendChild(btn('reset', () => { stage = 0; draw(); }));
  draw();
  return () => { host.removeChild(wrap); };
}

// ------------------------------------------------------------------- 2. valley

function animValley(host: HTMLElement) {
  const wrap = mk('div', 'tanim');
  const controls = mk('div', 'tcontrols');
  const kap = sliderRow('κ (steepness ratio)', 1, 50, 1, 1);
  const lr = sliderRow('learning rate', -3, -0.7, 0.01, -1.7);       // 10^v
  const readout = mk('div', 'tmsg', '');
  wrap.appendChild(controls); wrap.appendChild(kap.row); wrap.appendChild(lr.row);
  host.appendChild(wrap);
  const cv = makeCanvas(wrap, 300);
  wrap.appendChild(readout);

  let running = false, momentum = false;
  let px = -9, py = 2.4, vx = 0, vy = 0, steps = 0, raf = 0;
  const X0 = -10, X1 = 10, Y0 = -3.1, Y1 = 3.1;

  function reset() { px = -9; py = 2.4; vx = 0; vy = 0; steps = 0; }
  function toPx(x: number, yv: number, W: number, H: number) {
    return [ (x - X0) / (X1 - X0) * W, H - (yv - Y0) / (Y1 - Y0) * H ];
  }
  const path: number[][] = [];
  function stepGD(n: number) {
    const k = Number(kap.input.value);
    const eta = Math.pow(10, Number(lr.input.value));
    for (let i = 0; i < n; i++) {
      const gx = px, gy = k * py;
      if (momentum) {
        vx = 0.9 * vx - eta * gx; vy = 0.9 * vy - eta * gy;
        px += vx; py += vy;
      } else {
        px -= eta * gx; py -= eta * gy;
      }
      steps++;
      path.push([px, py]);
      if (path.length > 4000) path.shift();
      if (!isFinite(px) || !isFinite(py) || Math.abs(py) > 40) { running = false; break; }
    }
  }
  function draw() {
    const W = cv.width(), H = cv.height();
    const ctx = cv.ctx;
    ctx.clearRect(0, 0, W, H);
    const k = Number(kap.input.value);
    // contours
    ctx.strokeStyle = cssVar('--line');
    for (const c of [0.5, 2, 5, 12, 25, 45]) {
      const ax = Math.sqrt(2 * c), ay = Math.sqrt(2 * c / k);
      ctx.beginPath();
      for (let t = 0; t <= 64; t++) {
        const th = t / 64 * 2 * Math.PI;
        const p = toPx(ax * Math.cos(th), ay * Math.sin(th), W, H);
        if (t === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
    }
    // minimum
    const m = toPx(0, 0, W, H);
    ctx.fillStyle = cssVar('--ok');
    ctx.beginPath(); ctx.arc(m[0], m[1], 4, 0, 7); ctx.fill();
    // path
    ctx.strokeStyle = cssVar('--accent'); ctx.lineWidth = 1.6;
    ctx.beginPath();
    let started = false;
    for (const p of path) {
      const q = toPx(p[0], p[1], W, H);
      if (q[1] < -50 || q[1] > H + 50) { started = false; continue; }
      if (!started) { ctx.moveTo(q[0], q[1]); started = true; } else ctx.lineTo(q[0], q[1]);
    }
    ctx.stroke(); ctx.lineWidth = 1;
    const q = toPx(px, py, W, H);
    ctx.fillStyle = cssVar('--accent');
    ctx.beginPath(); ctx.arc(q[0], q[1], 5, 0, 7); ctx.fill();
    const fval = 0.5 * (px * px + k * py * py);
    let fstr = 'diverged!';
    if (isFinite(fval)) fstr = fval.toExponential(2);
    readout.textContent = 'steps: ' + steps + '   f = ' + fstr +
      '   lr = ' + Math.pow(10, Number(lr.input.value)).toFixed(3) + (momentum ? '   momentum ON (β=0.9)' : '   momentum off');
    lr.out.textContent = Math.pow(10, Number(lr.input.value)).toFixed(3);
    kap.out.textContent = kap.input.value;
  }
  function loop() {
    if (running) stepGD(4);
    draw();
    raf = requestAnimationFrame(loop);
  }
  const momBtn = btn('momentum: off', () => {
    momentum = !momentum;
    momBtn.textContent = momentum ? 'momentum: ON' : 'momentum: off';
    path.length = 0; reset();
  });
  controls.appendChild(btn('run / pause', () => { running = !running; }));
  controls.appendChild(btn('reset', () => { path.length = 0; reset(); }));
  controls.appendChild(momBtn);
  kap.input.addEventListener('input', () => { path.length = 0; reset(); });
  raf = requestAnimationFrame(loop);
  return () => { cancelAnimationFrame(raf); cv.cleanup(); host.removeChild(wrap); };
}

// ---------------------------------------------------------- 3. spectral bias MLP

function animSpectral(host: HTMLElement) {
  const N = 256, H = 64;
  const xs = new Float64Array(N);
  const ys = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    xs[i] = -Math.PI + (2 * Math.PI * i) / (N - 1);
    ys[i] = Math.sin(xs[i]) + 0.3 * Math.sin(15 * xs[i]);
  }
  // Fourier feature frequencies: seeded gaussian, sigma=8 (seed chosen so coverage
  // includes the 15-cycle band — verified numerically).
  const NF = 12;
  const B = new Float64Array(NF);
  { const r = rng(7); for (let j = 0; j < NF; j++) B[j] = gauss(r) * 8; }

  let fourier = false;
  let dIn = 1;
  let W1, b1, W2, b2v;
  let mW1, vW1, mb1, vb1, mW2, vW2, mb2 = 0, vb2 = 0;
  let adamT = 0, stepCount = 0, running = false, raf = 0;

  function features(x: number, out: Float64Array) {
    if (!fourier) { out[0] = x / Math.PI; return; }
    for (let j = 0; j < NF; j++) { out[2 * j] = Math.cos(B[j] * x); out[2 * j + 1] = Math.sin(B[j] * x); }
  }
  function init() {
    dIn = fourier ? 2 * NF : 1;
    const r = rng(42);
    W1 = new Float64Array(H * dIn); b1 = new Float64Array(H);
    W2 = new Float64Array(H); b2v = 0;
    const s1 = Math.sqrt(2 / (dIn + H)) * (5 / 3);
    for (let i = 0; i < W1.length; i++) W1[i] = gauss(r) * s1;
    for (let i = 0; i < H; i++) { b1[i] = (r() * 2 - 1) * 2; W2[i] = gauss(r) * Math.sqrt(2 / (H + 1)); }
    mW1 = new Float64Array(W1.length); vW1 = new Float64Array(W1.length);
    mb1 = new Float64Array(H); vb1 = new Float64Array(H);
    mW2 = new Float64Array(H); vW2 = new Float64Array(H);
    mb2 = 0; vb2 = 0; adamT = 0; stepCount = 0;
  }
  const feat = new Float64Array(24);
  const hid = new Float64Array(H);
  function forward(x: number) {
    features(x, feat);
    let out = b2v;
    for (let i = 0; i < H; i++) {
      let z = b1[i];
      const off = i * dIn;
      for (let j = 0; j < dIn; j++) z += W1[off + j] * feat[j];
      const h = Math.tanh(z);
      hid[i] = h;
      out += W2[i] * h;
    }
    return out;
  }
  function adam(p: Float64Array, g: Float64Array, m: Float64Array, v: Float64Array, lr: number, t: number) {
    const b1c = 1 - Math.pow(0.9, t), b2c = 1 - Math.pow(0.999, t);
    for (let i = 0; i < p.length; i++) {
      m[i] = 0.9 * m[i] + 0.1 * g[i];
      v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
      p[i] -= lr * (m[i] / b1c) / (Math.sqrt(v[i] / b2c) + 1e-8);
    }
  }
  const gW1 = new Float64Array(64 * 24); const gb1 = new Float64Array(H); const gW2 = new Float64Array(H);
  function trainSteps(n: number) {
    const lr = 3e-3;
    for (let s = 0; s < n; s++) {
      gW1.fill(0); gb1.fill(0); gW2.fill(0);
      let gb2 = 0;
      for (let i = 0; i < N; i++) {
        const pred = forward(xs[i]);
        const e = (2 / N) * (pred - ys[i]);
        gb2 += e;
        for (let k = 0; k < H; k++) {
          gW2[k] += e * hid[k];
          const dh = e * W2[k] * (1 - hid[k] * hid[k]);
          gb1[k] += dh;
          const off = k * dIn;
          for (let j = 0; j < dIn; j++) gW1[off + j] += dh * feat[j];
        }
        // NOTE: feat/hid hold the values of THIS sample because forward() just ran.
      }
      adamT++;
      adam(W1, gW1.subarray(0, W1.length), mW1, vW1, lr, adamT);
      adam(b1, gb1, mb1, vb1, lr, adamT);
      adam(W2, gW2, mW2, vW2, lr, adamT);
      mb2 = 0.9 * mb2 + 0.1 * gb2; vb2 = 0.999 * vb2 + 0.001 * gb2 * gb2;
      b2v -= lr * (mb2 / (1 - Math.pow(0.9, adamT))) / (Math.sqrt(vb2 / (1 - Math.pow(0.999, adamT))) + 1e-8);
      stepCount++;
    }
  }
  function amplitude(k: number) {
    let s = 0;
    for (let i = 0; i < N; i++) s += forward(xs[i]) * Math.sin(k * xs[i]);
    return (2 / N) * s;
  }

  const wrap = mk('div', 'tanim');
  const controls = mk('div', 'tcontrols');
  const readout = mk('div', 'tmsg', '');
  wrap.appendChild(controls);
  host.appendChild(wrap);
  const cv = makeCanvas(wrap, 320);
  wrap.appendChild(readout);

  function draw() {
    const W = cv.width(), Hc = cv.height(), ctx = cv.ctx;
    ctx.clearRect(0, 0, W, Hc);
    const plotH = Hc - 78;
    const toX = (x: number) => (x + Math.PI) / (2 * Math.PI) * (W - 20) + 10;
    const toY = (v: number) => plotH / 2 - v * (plotH / 3.4);
    ctx.strokeStyle = cssVar('--line'); ctx.beginPath(); ctx.moveTo(10, toY(0)); ctx.lineTo(W - 10, toY(0)); ctx.stroke();
    // target
    ctx.strokeStyle = cssVar('--fg-faint'); ctx.setLineDash([4, 4]); ctx.beginPath();
    for (let i = 0; i < N; i++) { const px = toX(xs[i]), py = toY(ys[i]); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
    ctx.stroke(); ctx.setLineDash([]);
    // prediction
    ctx.strokeStyle = cssVar('--accent'); ctx.lineWidth = 1.8; ctx.beginPath();
    for (let i = 0; i < N; i++) { const px = toX(xs[i]), py = toY(forward(xs[i])); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
    ctx.stroke(); ctx.lineWidth = 1;
    // amplitude bars
    const a1 = amplitude(1), a15 = amplitude(15);
    const bars = [ { label: 'k = 1 (target 1.0)', v: a1, t: 1.0 }, { label: 'k = 15 (target 0.3)', v: a15, t: 0.3 } ];
    const by = Hc - 58;
    for (let i = 0; i < 2; i++) {
      const bx = 20 + i * (W / 2);
      const bw = W / 2 - 60;
      ctx.fillStyle = cssVar('--bg-sunken'); ctx.fillRect(bx, by, bw, 14);
      const frac = Math.max(0, Math.min(1.15, bars[i].v / bars[i].t));
      let barCol = cssVar('--accent');
      if (frac > 0.85) barCol = cssVar('--ok');
      ctx.fillStyle = barCol;
      ctx.fillRect(bx, by, bw * Math.min(1, frac), 14);
      ctx.strokeStyle = cssVar('--line-strong'); ctx.strokeRect(bx, by, bw, 14);
      ctx.fillStyle = cssVar('--fg-dim'); ctx.font = '11px ' + cssVar('--sans');
      ctx.fillText(bars[i].label + '   recovered: ' + bars[i].v.toFixed(3), bx, by + 30);
    }
    readout.textContent = 'step ' + stepCount + '   ·   features: ' + (fourier ? 'Fourier (24-dim, σ=8)' : 'plain x (1-dim)') +
      '   ·   dashed = target, orange = network';
  }
  function loop() { if (running) trainSteps(20); draw(); raf = requestAnimationFrame(loop); }
  const fBtn = btn('Fourier features: off', () => {
    fourier = !fourier;
    fBtn.textContent = fourier ? 'Fourier features: ON' : 'Fourier features: off';
    init();
  });
  controls.appendChild(btn('run / pause', () => { running = !running; }));
  controls.appendChild(btn('restart', () => init()));
  controls.appendChild(fBtn);
  init();
  raf = requestAnimationFrame(loop);
  return () => { cancelAnimationFrame(raf); cv.cleanup(); host.removeChild(wrap); };
}

// --------------------------------------------------------------- 4. live PINN

function animPinnLive(host: HTMLElement) {
  const H = 32, NC = 64, PI = Math.PI;
  const xc = new Float64Array(NC);
  for (let i = 0; i < NC; i++) xc[i] = -1 + (2 * (i + 0.5)) / NC;
  // parameters: w[H], b[H], v[H], c  (97 total)
  const P = 3 * H + 1;
  let p = new Float64Array(P);
  let m = new Float64Array(P), vv = new Float64Array(P), adamT = 0;
  let hard = false, lambda = 1.0, running = false, steps = 0, raf = 0;

  function init() {
    const r = rng(3);
    p = new Float64Array(P);
    for (let i = 0; i < H; i++) {
      p[i] = gauss(r) * 1.2;                 // w
      p[H + i] = (r() * 2 - 1) * 1.5;        // b
      p[2 * H + i] = gauss(r) * 0.25;        // v
    }
    p[3 * H] = 0;
    m = new Float64Array(P); vv = new Float64Array(P); adamT = 0; steps = 0;
  }
  // raw net N(x), N'(x), N''(x) — closed-form for one hidden layer
  function netAll(x: number, out: Float64Array, pp: Float64Array) {
    let n0 = pp[3 * H], n1 = 0, n2 = 0;
    for (let i = 0; i < H; i++) {
      const w = pp[i], b = pp[H + i], v = pp[2 * H + i];
      const t = Math.tanh(w * x + b);
      const s = 1 - t * t;
      n0 += v * t;
      n1 += v * w * s;
      n2 += v * w * w * (-2 * t * s);
    }
    out[0] = n0; out[1] = n1; out[2] = n2;
  }
  const tmp = new Float64Array(3);
  function uAndUxx(x: number, pp: Float64Array) {
    netAll(x, tmp, pp);
    if (!hard) return [tmp[0], tmp[2]];
    const g = 1 - x * x;             // u = g·N ; u'' = g·N'' − 4x·N' − 2N
    return [g * tmp[0], g * tmp[2] - 4 * x * tmp[1] - 2 * tmp[0]];
  }
  function loss(pp: Float64Array) {
    let lr_ = 0;
    for (let i = 0; i < NC; i++) {
      const r_ = uAndUxx(xc[i], pp);
      const res = r_[1] + PI * PI * Math.sin(PI * xc[i]);
      lr_ += res * res;
    }
    lr_ /= NC;
    if (hard) return lr_;
    netAll(-1, tmp, pp); const uL = tmp[0];
    netAll(1, tmp, pp); const uR = tmp[0];
    return lr_ + lambda * (uL * uL + uR * uR);
  }
  const grad = new Float64Array(P);
  function trainSteps(n: number) {
    const lr = 8e-3;
    for (let s = 0; s < n; s++) {
      for (let i = 0; i < P; i++) {
        const h = 1e-6 * (1 + Math.abs(p[i]));
        const old = p[i];
        p[i] = old + h; const lp = loss(p);
        p[i] = old - h; const lm = loss(p);
        p[i] = old;
        grad[i] = (lp - lm) / (2 * h);
      }
      adamT++;
      const b1c = 1 - Math.pow(0.9, adamT), b2c = 1 - Math.pow(0.999, adamT);
      for (let i = 0; i < P; i++) {
        m[i] = 0.9 * m[i] + 0.1 * grad[i];
        vv[i] = 0.999 * vv[i] + 0.001 * grad[i] * grad[i];
        p[i] -= lr * (m[i] / b1c) / (Math.sqrt(vv[i] / b2c) + 1e-8);
      }
      steps++;
    }
  }
  function relL2() {
    let num = 0, den = 0;
    for (let i = 0; i <= 128; i++) {
      const x = -1 + (2 * i) / 128;
      const ue = Math.sin(PI * x);
      const up = uAndUxx(x, p)[0];
      num += (up - ue) * (up - ue); den += ue * ue;
    }
    return Math.sqrt(num / den);
  }

  const wrap = mk('div', 'tanim');
  const controls = mk('div', 'tcontrols');
  const lam = sliderRow('λ (boundary weight, soft mode)', -1, 2, 0.05, 0);   // 10^v
  const readout = mk('div', 'tmsg', '');
  wrap.appendChild(controls); wrap.appendChild(lam.row);
  host.appendChild(wrap);
  const cv = makeCanvas(wrap, 300);
  wrap.appendChild(readout);

  function draw() {
    lambda = Math.pow(10, Number(lam.input.value));
    lam.out.textContent = lambda.toFixed(1);
    const W = cv.width(), Hc = cv.height(), ctx = cv.ctx;
    ctx.clearRect(0, 0, W, Hc);
    const toX = (x: number) => (x + 1) / 2 * (W - 24) + 12;
    const toY = (v: number) => Hc / 2 - v * (Hc / 3.1);
    ctx.strokeStyle = cssVar('--line');
    ctx.beginPath(); ctx.moveTo(12, toY(0)); ctx.lineTo(W - 12, toY(0)); ctx.stroke();
    // exact (dashed)
    ctx.strokeStyle = cssVar('--fg-faint'); ctx.setLineDash([4, 4]); ctx.beginPath();
    for (let i = 0; i <= 128; i++) { const x = -1 + (2 * i) / 128; const px = toX(x), py = toY(Math.sin(PI * x)); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
    ctx.stroke(); ctx.setLineDash([]);
    // prediction
    ctx.strokeStyle = cssVar('--accent'); ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i <= 128; i++) { const x = -1 + (2 * i) / 128; const px = toX(x), py = toY(uAndUxx(x, p)[0]); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
    ctx.stroke(); ctx.lineWidth = 1;
    // collocation dots coloured by |residual|
    for (let i = 0; i < NC; i++) {
      const r_ = uAndUxx(xc[i], p);
      const res = Math.abs(r_[1] + PI * PI * Math.sin(PI * xc[i]));
      const t = Math.max(0, Math.min(1, Math.log10(res + 1e-6) / 2 + 1));   // ~[1e-6..1e2] -> [0..1]
      let col = cssVar('--ok');
      if (t > 0.66) col = cssVar('--accent');
      else if (t > 0.33) col = cssVar('--mark');
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(toX(xc[i]), toY(0), 3, 0, 7); ctx.fill();
    }
    // boundary markers
    ctx.fillStyle = cssVar('--fg-dim');
    ctx.beginPath(); ctx.arc(toX(-1), toY(0), 4, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.arc(toX(1), toY(0), 4, 0, 7); ctx.fill();
    const L = loss(p);
    readout.textContent = (hard ? 'HARD BC: u = (1−x²)·N(x) — no λ, nothing to balance' : 'SOFT BC, λ = ' + lambda.toFixed(1)) +
      '   ·   step ' + steps + '   ·   loss ' + L.toExponential(2) + '   ·   true rel-L2 error ' + relL2().toExponential(2);
  }
  function loop() { if (running) trainSteps(4); draw(); raf = requestAnimationFrame(loop); }
  const modeBtn = btn('mode: soft BC', () => {
    hard = !hard;
    modeBtn.textContent = hard ? 'mode: HARD BC' : 'mode: soft BC';
    lam.row.style.opacity = hard ? '0.35' : '1';
    init();
  });
  controls.appendChild(btn('train / pause', () => { running = !running; }));
  controls.appendChild(btn('restart', () => init()));
  controls.appendChild(modeBtn);
  init();
  raf = requestAnimationFrame(loop);
  return () => { cancelAnimationFrame(raf); cv.cleanup(); host.removeChild(wrap); };
}

// --------------------------------------------------- 5-7. measured beta charts

function animBetaCurves(host: HTMLElement) {
  const wrap = mk('div', 'tanim');
  const controls = mk('div', 'tcontrols');
  const readout = mk('div', 'tmsg', 'Measured in this repo: ' + BETA_DATA.config.net + ', ' + BETA_DATA.config.steps_per_stage + ' Adam steps, float64 CPU, seed 0.');
  wrap.appendChild(controls);
  host.appendChild(wrap);
  const cv = makeCanvas(wrap, 300);
  wrap.appendChild(readout);
  let t = 0, playing = false, raf = 0;
  const betaList = BETA_DATA.betas;
  const total = betaList.length * 1.0;
  function draw() {
    const W = cv.width(), Hc = cv.height(), ctx = cv.ctx;
    ctx.clearRect(0, 0, W, Hc);
    const mL = 46, mB = 30, mT = 12, mR = 12;
    const yOf = (e: number) => { const lo = Math.log10(0.02), hi = Math.log10(1.2); return mT + (hi - Math.log10(Math.max(e, 0.02))) / (hi - lo) * (Hc - mT - mB); };
    const xOf = (s: number) => mL + s / 3000 * (W - mL - mR);
    ctx.strokeStyle = cssVar('--line'); ctx.fillStyle = cssVar('--fg-faint'); ctx.font = '10.5px ' + cssVar('--mono');
    for (const g of [1, 0.3, 0.1, 0.03]) {
      ctx.beginPath(); ctx.moveTo(mL, yOf(g)); ctx.lineTo(W - mR, yOf(g)); ctx.stroke();
      ctx.fillText((g * 100).toFixed(0) + '%', 6, yOf(g) + 3);
    }
    ctx.fillText('Adam steps →', W - 100, Hc - 8);
    ctx.fillText('relative L2 error (log)', 6, 12);
    const cols = [cssVar('--ok'), '#7a9f35', cssVar('--mark'), cssVar('--accent')];
    for (let bi = 0; bi < betaList.length; bi++) {
      const frac = Math.max(0, Math.min(1, t - bi));
      if (frac <= 0) continue;
      const curve = BETA_DATA.vanilla_curves[String(betaList[bi])];
      const nPts = Math.max(2, Math.floor(curve.length * frac));
      ctx.strokeStyle = cols[bi] || cssVar('--accent'); ctx.lineWidth = 2; ctx.beginPath();
      for (let i = 0; i < nPts; i++) {
        const px = xOf(curve[i].step), py = yOf(curve[i].err);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke(); ctx.lineWidth = 1;
      const last = curve[nPts - 1];
      ctx.fillStyle = cols[bi] || cssVar('--accent');
      ctx.font = '12px ' + cssVar('--sans');
      ctx.fillText('β=' + betaList[bi] + (frac >= 1 ? '  →  ' + (BETA_DATA.vanilla_final[bi] * 100).toFixed(1) + '%' : ''), xOf(last.step) - 60, yOf(last.err) - 8);
    }
  }
  function loop() { if (playing && t < total) t += 0.02; draw(); raf = requestAnimationFrame(loop); }
  controls.appendChild(btn('▶ play', () => { playing = true; if (t >= total) t = 0; }));
  controls.appendChild(btn('reset', () => { t = 0; playing = false; }));
  raf = requestAnimationFrame(loop);
  return () => { cancelAnimationFrame(raf); cv.cleanup(); host.removeChild(wrap); };
}

function heat(ctx: CanvasRenderingContext2D, u: number[][], x0: number, y0: number, w: number, h: number, title: string) {
  const nx = u.length, nt = u[0].length;
  const cw = w / nt, ch = h / nx;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nt; j++) {
      const v = Math.max(-1, Math.min(1, u[i][j]));
      // diverging colormap: v=-1 blue, v=0 near-white, v=+1 warm red
      let rr = 0, gg = 0, bb = 0;
      if (v >= 0) {
        rr = Math.round(235 - v * 30);
        gg = Math.round(225 - v * 130);
        bb = Math.round(215 - v * 150);
      } else {
        rr = Math.round(235 + v * 140);
        gg = Math.round(225 + v * 90);
        bb = Math.round(235 - v * 8);
      }
      ctx.fillStyle = 'rgb(' + rr + ',' + gg + ',' + bb + ')';
      ctx.fillRect(x0 + j * cw, y0 + (nx - 1 - i) * ch, cw + 0.6, ch + 0.6);
    }
  }
  ctx.strokeStyle = cssVar('--line-strong'); ctx.strokeRect(x0, y0, w, h);
  ctx.fillStyle = cssVar('--fg-dim'); ctx.font = '12px ' + cssVar('--sans');
  ctx.fillText(title, x0, y0 - 6);
}

function animBetaFields(host: HTMLElement) {
  const wrap = mk('div', 'tanim');
  host.appendChild(wrap);
  const cv = makeCanvas(wrap, 260);
  const note = mk('div', 'tmsg', 'u(x, t) at β = 30 — x vertical, t horizontal. Left: the exact travelling wave. Right: what the vanilla PINN converged to (rel-L2 ' + (BETA_DATA.vanilla_final[3] * 100).toFixed(1) + '%).');
  wrap.appendChild(note);
  function draw() {
    const W = cv.width(), Hc = cv.height(), ctx = cv.ctx;
    ctx.clearRect(0, 0, W, Hc);
    const w = (W - 60) / 2, h = Hc - 40;
    heat(ctx, BETA_DATA.fields.exact, 12, 26, w, h, 'exact: sin(x − 30t)');
    heat(ctx, BETA_DATA.fields.vanilla, w + 46, 26, w, h, 'vanilla PINN, 3000 steps');
  }
  draw();
  const onR = () => draw();
  window.addEventListener('resize', onR);
  return () => { window.removeEventListener('resize', onR); cv.cleanup(); host.removeChild(wrap); };
}

function animBetaRescue(host: HTMLElement) {
  const wrap = mk('div', 'tanim');
  const controls = mk('div', 'tcontrols');
  wrap.appendChild(controls);
  host.appendChild(wrap);
  const cv = makeCanvas(wrap, 300);
  const note = mk('div', 'tmsg', 'Left: final rel-L2 at β = 30 for the three strategies (same architecture). Right: the time-marched solution field — compare with the exact field two steps back.');
  wrap.appendChild(note);
  let t = 0, playing = false, raf = 0;
  const bars = [
    { label: 'vanilla', v: BETA_DATA.vanilla_final[3] },
    { label: 'curriculum (small budget)', v: BETA_DATA.curriculum_final_beta30 },
    { label: 'time-marching, ' + BETA_DATA.marching_windows + ' windows', v: BETA_DATA.marching_final_beta30 },
  ];
  function draw() {
    const W = cv.width(), Hc = cv.height(), ctx = cv.ctx;
    ctx.clearRect(0, 0, W, Hc);
    const bw = W * 0.44, fh = Hc - 46;
    // bars (log scale)
    const yOf = (e: number) => { const lo = Math.log10(0.015), hi = Math.log10(1.2); return 20 + (hi - Math.log10(Math.max(e, 0.015))) / (hi - lo) * (fh - 20); };
    ctx.strokeStyle = cssVar('--line');
    ctx.fillStyle = cssVar('--fg-faint'); ctx.font = '10.5px ' + cssVar('--mono');
    for (const g of [1, 0.3, 0.1, 0.03]) {
      ctx.beginPath(); ctx.moveTo(34, yOf(g)); ctx.lineTo(bw, yOf(g)); ctx.stroke();
      ctx.fillText((g * 100).toFixed(0) + '%', 2, yOf(g) + 3);
    }
    const colW = (bw - 50) / 3;
    for (let i = 0; i < 3; i++) {
      const frac = Math.max(0, Math.min(1, t * 3 - i));
      const vShown = Math.pow(10, Math.log10(1.0) + (Math.log10(Math.max(bars[i].v, 0.015)) - 0) * frac);
      const x = 40 + i * colW;
      const yTop = yOf(frac <= 0 ? 1.0 : vShown);
      let rbCol = cssVar('--accent');
      if (i === 2) rbCol = cssVar('--ok');
      ctx.fillStyle = rbCol;
      ctx.globalAlpha = i === 1 ? 0.65 : 1;
      ctx.fillRect(x, yTop, colW - 16, 20 + fh - 20 - yTop);
      ctx.globalAlpha = 1;
      ctx.fillStyle = cssVar('--fg-dim'); ctx.font = '11px ' + cssVar('--sans');
      if (frac >= 1) ctx.fillText((bars[i].v * 100).toFixed(1) + '%', x, yTop - 5);
    }
    ctx.fillStyle = cssVar('--fg-faint'); ctx.font = '10.5px ' + cssVar('--sans');
    ctx.fillText('vanilla', 40, Hc - 26);
    ctx.fillText('curriculum', 40 + colW, Hc - 26);
    ctx.fillText('marching', 40 + 2 * colW, Hc - 26);
    // marched field
    heat(cv.ctx, BETA_DATA.march_field.u, bw + 40, 26, W - bw - 56, fh - 20, 'time-marched PINN (rel-L2 ' + (BETA_DATA.marching_final_beta30 * 100).toFixed(1) + '%)');
  }
  function loop() { if (playing && t < 1) t += 0.012; draw(); raf = requestAnimationFrame(loop); }
  controls.appendChild(btn('▶ play', () => { playing = true; if (t >= 1) t = 0; }));
  raf = requestAnimationFrame(loop);
  return () => { cancelAnimationFrame(raf); cv.cleanup(); host.removeChild(wrap); };
}

// ------------------------------------------------------------------- registry

export const ANIMS = {
  'backprop-graph': animBackpropGraph,
  'valley': animValley,
  'spectral': animSpectral,
  'pinn-live': animPinnLive,
  'beta-curves': animBetaCurves,
  'beta-fields': animBetaFields,
  'beta-rescue': animBetaRescue,
};

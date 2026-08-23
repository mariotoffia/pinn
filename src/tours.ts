// Guided tours: Brilliant-style stepped lessons layered over the chapters.
//
// Pure data. Each step has a short body (HTML), and optionally: an animation id
// (implemented in touranim.ts), a quiz (Next is never blocked - checking is for
// learning, not gating), and a notebook block that routes to a local marimo lab,
// a starter script, or an external Colab-ready notebook. Notebook blocks always
// carry three fields the reader needs before running anything: what it does,
// what knowledge it rests on, and what a successful run looks like.

export const TOURS = [
  {
    slug: '02-neural-networks',
    title: 'See backprop happen',
    minutes: 12,
    steps: [
      {
        title: 'What you will do here',
        body: '<p>In about ten minutes you will watch a tiny network compute — forward — and then watch the chain rule run <em>backwards</em> through it, number by number. At the end, <code>loss.backward()</code> will no longer be magic. Nothing to install; everything runs on this page.</p><p>You can always press <b>Next</b>. Stuck is not a state this tour allows.</p>',
      },
      {
        title: 'One neuron, three operations',
        body: '<p>The whole demo is one neuron: multiply the input by a weight, add a bias, squash with <code>tanh</code>, and measure the squared error against a target:</p><p><code>z = w·x + b</code> &nbsp;→&nbsp; <code>a = tanh(z)</code> &nbsp;→&nbsp; <code>L = (a − y)²</code></p><p>Below is that program drawn as a graph. Every box is a value; every arrow is an operation. This picture <em>is</em> the "computation graph" every deep-learning framework builds silently.</p>',
        anim: 'backprop-graph',
      },
      {
        title: 'Run the forward pass',
        body: '<p>Press <b>step ▸</b> above until the green numbers reach the loss. Each press executes one operation, left to right. Notice there is nothing statistical here — it is plain arithmetic on concrete numbers (<code>x = 2</code>, <code>w = −1.5</code>, <code>b = 1</code>, target <code>y = −0.5</code>).</p>',
        anim: 'backprop-graph',
      },
      {
        title: 'Quick check',
        quiz: {
          q: 'If we nudge the weight w a tiny bit, which values change?',
          options: [
            'Only z',
            'z, a and L — everything downstream of w',
            'Everything, including the input x',
          ],
          correct: 1,
          explain: 'A change flows only forward along the arrows: w feeds z, z feeds a, a feeds L. The input x is upstream of w, so it is untouched. Backprop is exactly this observation, run in reverse: L is sensitive to w through that same chain.',
        },
        body: '',
      },
      {
        title: 'Now run it backwards',
        body: '<p>Keep pressing <b>step ▸</b>. After the forward pass, the orange numbers appear from the right: <code>∂L/∂a</code>, then <code>∂L/∂z</code>, then <code>∂L/∂w</code> and <code>∂L/∂b</code>. Each one is the previous gradient times one <em>local</em> derivative — the chain rule, applied once per arrow. That is the entire backpropagation algorithm.</p>',
        anim: 'backprop-graph',
      },
      {
        title: 'The step that matters for PINNs',
        body: '<p>Look at the last orange number: <code>∂L/∂x</code> — the gradient with respect to the <em>input</em>. The graph does not care that x is "data" and w is "a parameter"; it can differentiate with respect to anything that flows in.</p><p><b>That indifference is the door to physics-informed networks.</b> A PINN differentiates the network with respect to its input coordinates (x, t) to build the PDE terms. Same machinery, different target. Chapter 03 walks through it.</p>',
        anim: 'backprop-graph',
      },
      {
        title: 'Do it with your own hands',
        body: '<p>The animation showed one neuron. The exercise below scales the same idea to a full multi-layer network — with <em>you</em> writing the backward pass.</p>',
        notebook: {
          kind: 'script',
          label: 'starter/scripts/01_mlp_from_scratch.py',
          run: 'cd starter && python scripts/01_mlp_from_scratch.py',
          what: 'Trains a small NumPy MLP on a 1D regression problem with a completely hand-written backward pass — no autograd anywhere — then checks every gradient against finite differences.',
          foundations: 'This animation, plus Nielsen Ch. 1–2 (the four backprop equations). If the equations are still shaky, do Nielsen first with a pen.',
          expect: 'Gradient check vs finite differences around 3.4e-08 relative, final MSE around 8.0e-06. If your own re-implementation matches those orders of magnitude, your backward pass is right.',
        },
      },
      {
        title: 'Prefer a guided video + notebook instead?',
        body: '<p>If you want the same lesson at full depth, with a master teacher typing every line:</p>',
        notebook: {
          kind: 'colab',
          label: 'Karpathy — micrograd (Zero to Hero, Lecture 1)',
          url: 'https://github.com/karpathy/nn-zero-to-hero',
          run: 'Open the lecture notebook from the repo in Colab, or just follow the video typing locally.',
          what: 'Builds a ~150-line reverse-mode autodiff engine from nothing, live, in 2h25m — the exact machinery this tour animated.',
          foundations: 'Comfort with Python classes and the chain rule for one variable. Nothing else.',
          expect: 'You end with your own working autograd engine whose gradients match PyTorch on the same tiny graphs. Keep it — Chapter 03 asks you to extend the idea.',
        },
      },
      {
        title: 'Done',
        body: '<p>You have now <em>seen</em> the two passes that all of deep learning runs on. Read §2.2 of this chapter next — with a pen — and then move to Chapter 03, which is where this path becomes about physics.</p>',
      },
    ],
  },

  {
    slug: '03-autodiff',
    title: 'grad(grad(u)) — the hinge',
    minutes: 12,
    steps: [
      {
        title: 'What you will do here',
        body: '<p>This tour makes the one mental shift the whole path depends on: from "differentiate the loss with respect to the <em>weights</em>" to "differentiate the network with respect to its <em>inputs</em> — twice — and then differentiate <em>that</em> with respect to the weights."</p><p>Ten minutes, three quiz checks, one local lab at the end.</p>',
      },
      {
        title: 'Where the last tour left off',
        body: '<p>The backprop animation in Chapter 02 ended on a deliberately strange number: <code>∂L/∂x</code>, the gradient with respect to the <em>input</em>. Press <b>step ▸</b> through to the end again if you like — it is the same graph.</p><p>In ordinary ML that number is a curiosity. In a PINN it is the product: <code>∂u/∂x</code> and <code>∂²u/∂x²</code> are the ingredients of the physics.</p>',
        anim: 'backprop-graph',
      },
      {
        title: 'The nesting',
        body: '<p>A second derivative by autodiff means: build the graph for <code>∂u/∂x</code>, then run backprop <em>through that graph</em>. The graph of the derivative is itself differentiable — as long as you keep it alive. In PyTorch that is one flag:</p><p><code>u_x&nbsp;&nbsp;= grad(u, x, create_graph=True)</code><br><code>u_xx = grad(u_x, x, create_graph=True)</code></p><p>The cost is real: the graph for <code>u_xx</code> is roughly twice the size of the graph for <code>u_x</code>. That is why 4th-order PDEs hurt.</p>',
      },
      {
        title: 'Check: the silent failure',
        quiz: {
          q: 'What happens if you forget create_graph=True on the first grad call?',
          options: [
            'PyTorch raises an error immediately',
            'Nothing — the flag is just an optimisation',
            'The weight gradient comes out wrong or zero, with no error',
          ],
          correct: 2,
          explain: 'The intermediate derivative becomes a dead end with no history, and the final .backward() silently computes gradients that ignore the physics term. It is the most common first PINN bug — and the loss still goes down, which is what makes it dangerous.',
        },
        body: '',
      },
      {
        title: 'Two directions of autodiff',
        body: '<p>Reverse mode (backprop) costs one sweep per <em>output</em>. Forward mode costs one sweep per <em>input</em>. A PINN has both shapes inside one loss:</p><p>• PDE derivatives: 1–4 input coordinates → <b>forward mode is cheap</b><br>• Weight gradient: thousands of weights, one scalar loss → <b>reverse mode is right</b></p><p>That is exactly the combination <code>jacfwd(jacrev(u))</code> builds — and what chained <code>autograd.grad</code> calls fail to exploit.</p>',
      },
      {
        title: 'Check: pick the mode',
        quiz: {
          q: 'You need ∂u/∂x, ∂u/∂y, ∂u/∂z, ∂u/∂t of a scalar network output. Which direction is efficient?',
          options: [
            'Reverse mode — it is always fastest',
            'Forward mode — only 4 inputs, so 4 cheap sweeps',
            'Neither — use finite differences',
          ],
          correct: 1,
          explain: 'Four inputs means four forward-mode sweeps, each about the cost of a forward pass. Reverse mode would also work but composes worse for the second derivative. Finite differences would throw away the one thing PINNs get for free: exactness.',
        },
        body: '',
      },
      {
        title: 'Exactness — the strongest true claim',
        body: '<p>These derivatives are <b>exact to machine precision</b>. Not approximated on a grid, no step size, no truncation error. All the error in a PINN comes from the network being imperfect and the optimiser stopping short — never from the derivatives themselves.</p><p>Your check, always: a central finite difference <code>(u(x+h) − 2u(x) + u(x−h))/h²</code> with <code>h = 1e-5</code> should agree with autodiff to about six digits in float64. If it does not, your graph is wrong.</p>',
      },
      {
        title: 'The local lab',
        body: '<p>Time to run the hinge exercise for real — the Burgers residual, computed by autodiff, with every claim above verified on your machine.</p>',
        notebook: {
          kind: 'marimo',
          label: 'notebooks/02_autodiff_lab.py',
          run: 'uvx marimo edit notebooks/02_autodiff_lab.py',
          what: 'An interactive lab: compute ∂u/∂x and ∂²u/∂x² of a tanh MLP, assemble the Burgers residual at thousands of random points, compare Idiom A (autograd.grad) with Idiom B (torch.func), and verify both against finite differences — with sliders for batch size and viscosity.',
          foundations: 'This tour, plus §3.1–3.4 of the chapter. You need PyTorch installed (Chapter 04 setup) — everything runs on CPU in seconds.',
          expect: 'Finite differences agree with autodiff to ~1e-06 relative; the two idioms agree to ~1e-16; every parameter gradient is finite. When you see those three numbers, you have written the core of a PINN.',
        },
      },
      {
        title: 'The Colab lane',
        body: '<p>Prefer a hosted notebook, or want the JAX view of the same ideas?</p>',
        notebook: {
          kind: 'colab',
          label: 'Dive into Deep Learning — Automatic Differentiation (runs in Colab)',
          url: 'https://d2l.ai/chapter_preliminaries/autograd.html',
          run: 'Every d2l.ai section has an "Open in Colab" button at the top — pick the PyTorch tab.',
          what: 'The autograd section of d2l.ai: builds up backward(), non-scalar outputs, detaching, and gradients through Python control flow — executable line by line.',
          foundations: 'Basic PyTorch tensors. It overlaps the start of this tour, then goes further into control flow.',
          expect: 'You can predict every printed gradient before running the cell. Then read the JAX Autodiff Cookbook (linked in §3.3) for the jacfwd/jacrev composition — even if you never write JAX.',
        },
      },
      {
        title: 'Done',
        body: '<p>Everything after this point in the path assumes fluency with <code>grad(grad(u))</code>. If the lab felt smooth, go straight to §3.6 and do the full exercise from a blank file. If not, watch the micrograd lecture from the Chapter 02 tour first — it is the best 2.5 hours this path can offer.</p>',
      },
    ],
  },

  {
    slug: '05-training-craft',
    title: 'Why training stalls',
    minutes: 15,
    steps: [
      {
        title: 'What you will do here',
        body: '<p>Two live experiments, in your browser: first you will <em>feel</em> ill-conditioning by steering gradient descent through a narrow valley; then you will watch a real network learn low frequencies first and leave the high frequency behind — spectral bias, the single most common reason PINNs fail.</p>',
      },
      {
        title: 'A round bowl is easy',
        body: '<p>Below: gradient descent on <code>f = ½(x² + κ·y²)</code>. With <b>κ = 1</b> the bowl is round — press <b>run</b> and watch it walk straight home. Nothing interesting. Now make it interesting: drag <b>κ</b> up to 30.</p>',
        anim: 'valley',
      },
      {
        title: 'A narrow valley is not',
        body: '<p>With κ large, the loss is steep across the valley and nearly flat along it — the definition of <b>ill-conditioned</b>. Try to fix it with the learning rate: too big and it zig-zags or explodes across the walls; small enough to be stable, and it crawls along the floor. There is <em>no good value</em>. That is the trap: the problem is the geometry, not your tuning.</p><p>Now switch <b>momentum</b> on. The zig-zags cancel, the along-valley speed accumulates, and it converges. This is the exact story of the Distill momentum article — and the mental model for every stalled PINN loss you will ever stare at.</p>',
        anim: 'valley',
      },
      {
        title: 'Check',
        quiz: {
          q: 'Your PINN loss falls fast, then flatlines at 1e-3 for thousands of steps. First hypothesis?',
          options: [
            'The network is too small — add layers',
            'The landscape is ill-conditioned — the optimiser is crawling along a valley floor',
            'The learning rate is too small — multiply it by 10',
          ],
          correct: 1,
          explain: 'Krishnapriyan et al. showed the same networks can represent the solutions easily — capacity is rarely the issue. The differential operator inside the loss makes the landscape stiff (Rathore et al., ICML 2024). Bigger networks make that worse, and a 10× learning rate usually finds the valley walls. The chapter-10 answers: better conditioning (non-dimensionalise, Fourier features) and curvature-aware optimisers (Adam → L-BFGS).',
        },
        body: '',
      },
      {
        title: 'Experiment two: watch spectral bias live',
        body: '<p>The target below is <code>sin(x) + 0.3·sin(15x)</code> — one slow wave carrying one fast ripple. The network is a real tanh MLP training in your browser. Press <b>run</b> and watch the two amplitude bars: the slow mode (k=1) is learned almost immediately; the fast mode (k=15) barely moves. This is <b>spectral bias</b>: networks learn low frequencies first, high frequencies last or never.</p>',
        anim: 'spectral',
      },
      {
        title: 'Now fix it',
        body: '<p>Switch <b>Fourier features</b> on and press <b>restart</b>. The inputs now pass through fixed random sines and cosines before the network — which turns "hard high frequency" into "easy low frequency" from the network\'s point of view. Watch the k=15 bar climb to 0.3.</p><p>This one toggle is Step 2 of the Chapter 10 recipe, and the same measured effect as <code>starter/scripts/03_spectral_bias.py</code>: without features the fast mode is recovered at 0.05 of its true 0.30 amplitude — never — and with them, at 0.3000.</p>',
        anim: 'spectral',
      },
      {
        title: 'Check',
        quiz: {
          q: 'Why does spectral bias hit PINNs harder than ordinary regression?',
          options: [
            'PINN networks are smaller than usual',
            'The loss is built from derivatives, and differentiation amplifies exactly the high-frequency error the network learns last',
            'Collocation points are random, so high frequencies are undersampled',
          ],
          correct: 1,
          explain: 'Differentiating multiplies each frequency component k by k (and k² for second derivatives) — so the part of the error the network is worst at dominates the residual. Classical iterative solvers are the exact mirror image: they kill high frequencies first, which is what makes multigrid work. §5.6 has the full story.',
        },
        body: '',
      },
      {
        title: 'The local lab',
        body: '',
        notebook: {
          kind: 'marimo',
          label: 'notebooks/03_spectral_bias_lab.py',
          run: 'uvx marimo edit notebooks/03_spectral_bias_lab.py',
          what: 'The same experiment with real PyTorch and full control: sliders for the fast frequency, its amplitude, the Fourier-feature scale σ and the training budget; live plots of the fit and the recovered amplitudes.',
          foundations: 'This tour. To understand *why* the fix works, read the Fourier Features paper (Tancik et al.) afterwards — the lab links the exact sections.',
          expect: 'Without features the fast-mode amplitude stalls far below its true value; with features it converges. Try σ = 0.5 and σ = 20 to see the too-blurry / too-noisy failure modes on either side of the σ ∈ [1,10] recommendation.',
        },
      },
      {
        title: 'The Colab lane',
        body: '',
        notebook: {
          kind: 'colab',
          label: 'Tancik et al. — Fourier Features demo (official Colab)',
          url: 'https://github.com/tancik/fourier-feature-networks',
          run: 'Open the demo notebook from the repo README — it has an Open-in-Colab badge.',
          what: 'The paper authors\' own 1D/2D fitting demo with and without the random Fourier mapping, on GPU if Colab grants one.',
          foundations: 'This tour is enough to read the demo; the NTK theory in the paper is optional on a first pass.',
          expect: 'The image-fitting example makes the effect unforgettable: without features the reconstruction is blurry at any training length; with them it is sharp. Thirty minutes, permanent intuition.',
        },
      },
      {
        title: 'Done',
        body: '<p>You have now seen — not read about — the two mechanisms behind most PINN training failure: ill-conditioning and spectral bias. Chapter 09 shows what they do to real physics problems, and Chapter 10 packages the fixes into a checklist. Next stop: §5.2, because the optimiser that exploits curvature (L-BFGS) is the other half of the story.</p>',
      },
    ],
  },

  {
    slug: '08-pinn-core',
    title: 'A PINN is born',
    minutes: 15,
    steps: [
      {
        title: 'What you will do here',
        body: '<p>You will train an actual physics-informed neural network, live, on this page — no install, no data, nothing but an equation. Along the way you will meet the three ideas that define the method: the <b>residual</b>, <b>collocation points</b>, and the <b>soft-vs-hard constraint</b> trade-off.</p>',
      },
      {
        title: 'The problem',
        body: '<p>We will solve a 1D Poisson equation on [−1, 1]:</p><p><code>u″(x) = −π²·sin(πx)</code>, &nbsp; with &nbsp; <code>u(−1) = u(1) = 0</code></p><p>The exact answer happens to be <code>sin(πx)</code> — but the network will never be told that. It will only ever be told how badly it <em>violates the equation</em>. That violation, <code>r(x) = u″(x) + π²·sin(πx)</code>, is called the <b>residual</b>, and it can be measured at any point without any data at all.</p>',
      },
      {
        title: 'Meet your network',
        body: '<p>Below: a real tanh network (one hidden layer, 32 units) with randomly initialised weights — the wiggly curve. The dots along the axis are <b>collocation points</b>: the places where we will measure the residual. Their colour shows |r| — bright means the physics is badly violated there. Press <b>train</b>.</p>',
        anim: 'pinn-live',
      },
      {
        title: 'What you are watching',
        body: '<p>Every training step: compute u″ at each dot by automatic differentiation (exactly — the second derivative of this network is a formula, not an approximation), square the residual, average, add a penalty for missing the boundary values, and take an optimiser step. The curve bends toward the hidden solution because <em>the equation leaves it nowhere else to go</em>.</p><p>Watch the two numbers: the <b>loss</b> (what the optimiser sees) and the <b>true error</b> (which it never sees — we can only print it because we happen to know the answer).</p>',
        anim: 'pinn-live',
      },
      {
        title: 'Break the balance',
        body: '<p>The loss has two parts: physics and boundary, weighted by λ. Drag <b>λ</b> down to 0.1, press <b>restart</b>, and watch: the physics term is happy, but the curve floats off the boundary values — a perfectly valid solution of the <em>equation</em>, just not of <em>our problem</em>. Now push λ to 100: the ends are pinned, and the interior learns more slowly.</p><p>There is no principled recipe for λ — that single fact generates half of Chapter 09 and several steps of Chapter 10.</p>',
        anim: 'pinn-live',
      },
      {
        title: 'Check',
        quiz: {
          q: 'With λ far too small, the trained network has a tiny residual loss but the wrong shape. Why?',
          options: [
            'The optimiser is broken',
            'Many functions satisfy the PDE; only the boundary conditions select ours — and we barely penalised missing them',
            'The collocation points were badly placed',
          ],
          correct: 1,
          explain: 'A differential equation alone has a whole family of solutions. The boundary/initial conditions pick one out. Soft penalties mean the network can trade a little boundary error for a lot of residual comfort — and it will. This is failure-by-design, and it is why the loss going down is never evidence of correctness (Chapter 09, §9.8).',
        },
        body: '',
      },
      {
        title: 'Now delete the problem',
        body: '<p>Flip the toggle to <b>hard BC</b> and restart. The network output is now wrapped as <code>u(x) = (1 − x²)·N(x)</code> — zero at the boundaries <em>by construction</em>. The boundary loss term is gone, λ is gone, and there is nothing left to balance. Watch it converge faster and cleaner.</p><p>This is the Lagaris trick from 1997, and Step 4 of the recipe: <b>every constraint you build in exactly is one loss term and one weight you never have to balance.</b></p>',
        anim: 'pinn-live',
      },
      {
        title: 'Check',
        quiz: {
          q: 'Why is the hard-constraint version not always used, if it is so clearly better here?',
          options: [
            'It is slower to evaluate',
            'Writing a wrapper that satisfies the conditions exactly gets hard on complex geometry — easy on [−1,1], research-grade on a turbine blade',
            'It only works for linear equations',
          ],
          correct: 1,
          explain: 'On an interval, (1−x²) does the job. On an arbitrary 3D domain you need an approximate distance function that vanishes exactly on the whole boundary — that is the Sukumar & Srivastava construction in Chapter 10, Step 4. Where you can build it, do; where you cannot, you are back to balancing λ.',
        },
        body: '',
      },
      {
        title: 'The real thing, on your machine',
        body: '<p>The browser demo hid one dimension and the time axis. The lab below is the genuine article — a damped-oscillator PINN with both constraint styles, side by side.</p>',
        notebook: {
          kind: 'marimo',
          label: 'notebooks/04_oscillator_pinn_lab.py',
          run: 'uvx marimo edit notebooks/04_oscillator_pinn_lab.py',
          what: 'Trains a PINN for the damped harmonic oscillator with a switch between soft and hard initial conditions and a λ slider — the exact experiment behind the measured numbers in starter/README.md.',
          foundations: 'This tour plus §8.5–8.7 (the residual, collocation, the combined loss). PyTorch on CPU; a run is a couple of minutes at the default budget.',
          expect: 'Hard IC beats the best soft-IC run: the starter kit measures 1.6e-4 (hard) vs 4.8e-4 (soft, λ=1) vs 1.1e-2 (soft, λ=100 — note the hand-tuned weight is *worse*). Reproducing that ordering is the lesson.',
        },
      },
      {
        title: 'The Colab lane',
        body: '',
        notebook: {
          kind: 'colab',
          label: 'Ben Moseley — harmonic oscillator PINN workshop (Colab badge in repo)',
          url: 'https://github.com/benmoseley/harmonic-oscillator-pinn-workshop',
          run: 'Open the student notebook via the repo\'s Open-in-Colab badge; the instructor version has the answers.',
          what: 'A from-scratch PyTorch PINN workshop — forward problem and inversion — with fill-in-the-blank cells, built by the FBPINNs author.',
          foundations: 'This tour is exactly the preparation it assumes. No PINN library involved, on purpose.',
          expect: 'A trained oscillator PINN that extrapolates where pure data-fitting fails, and (in part 2) a recovered physical parameter from noisy data — your first taste of the inverse problems PINNs are actually good at.',
        },
      },
      {
        title: 'Done',
        body: '<p>You have trained a PINN, broken it with a bad λ, and deleted the problem with a hard constraint. Now read §8.9 — the Burgers example written out in full — and then go straight to Chapter 09 before the excitement sets in.</p>',
      },
    ],
  },

  {
    slug: '09-failure-modes',
    title: 'Watch it fail — with real numbers',
    minutes: 12,
    steps: [
      {
        title: 'What you will do here',
        body: '<p>Every number in this tour was measured by running the code in this repository (CPU, float64, seed 0 — the config card is on the next step). No paper is being taken at its word. You will watch the cleanest failure in the PINN literature happen, then watch the fix this chapter calls the most reliable one actually work.</p>',
      },
      {
        title: 'The setup: an insultingly easy equation',
        body: '<p>1D convection: <code>u_t + β·u_x = 0</code> with <code>u(x,0) = sin(x)</code> and periodic boundaries. The exact solution is just the initial wave sliding right at speed β: <code>u = sin(x − βt)</code>. Nothing sharpens, nothing interacts — a first-year problem.</p><p>The runs: a 4×50 tanh network, 3000 Adam steps, 2560 collocation points, loss weights λ<sub>ic</sub> = λ<sub>bc</sub> = 100 — the standard vanilla recipe, trained from scratch at β = 1, 5, 15, 30.</p>',
      },
      {
        title: 'The failure, measured',
        body: '<p>Press <b>play</b>: each curve is the true relative L² error during training, one per β. At β = 1 the PINN is fine (3.2% error). At β = 30 — the <em>same solution shape, just faster</em> — it lands at <b>91% error</b>. The optimiser converges; the answer is garbage.</p>',
        anim: 'beta-curves',
      },
      {
        title: 'Check',
        quiz: {
          q: 'Is the β = 30 failure a capacity problem — is the network too small to represent sin(x − 30t)?',
          options: [
            'Yes — fast waves need bigger networks',
            'No — the same network fits the exact solution easily by plain regression; the optimisation of the residual loss is what fails',
            'Yes — float32 rounding prevents it',
          ],
          correct: 1,
          explain: 'This is Krishnapriyan et al.\'s decisive experiment: train the identical architecture by ordinary regression on the true solution and it fits with no trouble. The failure lives entirely in the soft-constrained residual optimisation — the landscape, not the hypothesis space. (And these runs are float64, so precision is not the culprit either.)',
        },
        body: '',
      },
      {
        title: 'What 91% error looks like',
        body: '<p>Below: the solution field u(x, t) as an image — exact on the left, the vanilla PINN\'s answer on the right. The network gets the first moments roughly right, then gives up on transporting the wave and smears toward something easy. It is trying to satisfy t = 1 before it has learned t = 0.1 — the anti-causal training of §9.3, visible.</p>',
        anim: 'beta-fields',
      },
      {
        title: 'The rescue, measured',
        body: '<p>Same architecture, same optimiser, same total budget shape — but now the time axis is split into 10 windows solved <em>in order</em>, each starting from the previous window\'s end state. Press <b>play</b>: global error at β = 30 drops from <b>91.4%</b> to <b>2.5%</b> — a 36× improvement, from nothing but respecting the arrow of time.</p><p>Full honesty, because that is the house style: a quick parameter-curriculum at the same small budget only managed 91% → 83% in our runs. The chapter\'s claim stands as written: <b>marching in time is the single most reliable practical fix</b> — curriculum helps, but needs a longer ladder and budget than this footnote-sized experiment gave it.</p>',
        anim: 'beta-rescue',
      },
      {
        title: 'Check',
        quiz: {
          q: 'Why does time-marching succeed where the all-at-once training failed?',
          options: [
            'It uses more collocation points overall',
            'Each window is a short-horizon, nearly-static problem — easy to optimise — and causality is enforced by construction, window by window',
            'Smaller windows mean smaller networks are needed',
          ],
          correct: 1,
          explain: 'Over a short window the solution barely moves, so the optimisation is benign; and because window k+1 starts from window k\'s answer, the network can never satisfy the future before the past. Causal weighting (§9.3) is the smooth version of the same idea inside one window.',
        },
        body: '',
      },
      {
        title: 'Reproduce, then go bigger',
        body: '<p>These tour numbers came from a miniature version of the sweep you are asked to run yourself in Week 10 of the roadmap. The full-strength version of the failure-and-fix story is the Burgers ablation:</p>',
        notebook: {
          kind: 'script',
          label: 'starter/scripts/06_pinn_burgers.py',
          run: 'cd starter && python scripts/06_pinn_burgers.py --plain   # then: --all',
          what: 'The canonical benchmark with every Chapter-10 fix behind a flag: Fourier features, RWF, causal weighting, gradient-norm balancing, Adam→L-BFGS, RAR.',
          foundations: 'Chapters 08–10. Run --plain first so you have a baseline number that is yours.',
          expect: 'Measured here: --plain 1.4e-2 → --all 4.7e-3 (three seeds tell you the variance; the paper\'s 6.7e-4 needs a much longer L-BFGS budget — see starter/README.md).',
        },
      },
      {
        title: 'The GPU lane',
        body: '<p>When you want the modern full-stack version of these fixes on genuinely hard problems:</p>',
        notebook: {
          kind: 'colab',
          label: 'jaxpi — the Expert\'s Guide reference implementation (GPU-only)',
          url: 'https://github.com/PredictiveIntelligenceLab/jaxpi',
          run: 'Clone in a Colab GPU session — the README is explicit that it targets CUDA; do not run it locally on the Mac.',
          what: 'The Predictive Intelligence Lab\'s JAX codebase implementing every trick in the Expert\'s Guide, with configs for Allen–Cahn, Kuramoto–Sivashinsky, lid-driven cavity, Navier–Stokes and more.',
          foundations: 'Chapter 10 read in full, and comfort reading JAX (the Cookbook from Chapter 03 is enough).',
          expect: 'Reproduced state-of-the-art PINN results — and a calibrated sense of how much machinery "state of the art" takes. Compare its Table-1 numbers with what your laptop achieved; the gap is the honest cost of the last digit.',
        },
      },
      {
        title: 'Done',
        body: '<p>You have now seen a PINN fail catastrophically on a trivial equation and be rescued by respecting causality — with numbers measured in this repository, not quoted from anywhere. Read §9.6 for where the 2024–2026 optimiser work takes this, and §9.8 for the paragraph to internalise. Then Chapter 10 turns all of it into a checklist.</p>',
      },
    ],
  },
];

export function tourFor(slug: string) {
  for (const t of TOURS) if (t.slug === slug) return t;
  return null;
}

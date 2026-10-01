/* ══════════════════════════════════════════════════════════════════
   Multimodal Flow — project page interactions
   Vanilla JS, no dependencies.
   ══════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  const $  = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.prototype.slice.call((r || document).querySelectorAll(s));
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ───────────────────────── 1. hero flow field ───────────────────────── */
  (function heroField() {
    const cv = $('#flowfield');
    if (!cv) return;
    const ctx = cv.getContext('2d');
    let W = 0, H = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const N = 140;
    const parts = [];
    const colors = ['rgba(150,190,240,', 'rgba(160,215,150,', 'rgba(185,175,225,', 'rgba(210,225,245,'];

    function resize() {
      const r = cv.parentElement.getBoundingClientRect();
      W = r.width; H = r.height;
      cv.width = W * dpr; cv.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#121824';
      ctx.fillRect(0, 0, W, H);
    }

    function seed() {
      parts.length = 0;
      for (let i = 0; i < N; i++) {
        parts.push({
          x: Math.random() * W,
          y: Math.random() * H,
          life: Math.random() * 220,
          c: colors[(Math.random() * colors.length) | 0],
          w: 0.5 + Math.random() * 1.1
        });
      }
    }

    // smooth pseudo-curl field: flows generally left→right, converging into bands
    function field(x, y, t) {
      const a = Math.sin(x * 0.0032 + t * 0.22) * 1.1
              + Math.cos(y * 0.0041 - t * 0.17) * 0.9
              + Math.sin((x + y) * 0.0017 + t * 0.11) * 0.6;
      return a * 0.55;
    }

    let tt = 0;
    function frame() {
      tt += 0.016;
      ctx.fillStyle = 'rgba(16,21,32,0.055)';
      ctx.fillRect(0, 0, W, H);

      for (let i = 0; i < parts.length; i++) {
        const p = parts[i];
        const ang = field(p.x, p.y, tt);
        const vx = Math.cos(ang) * 1.5 + 0.85;
        const vy = Math.sin(ang) * 1.5;
        const nx = p.x + vx, ny = p.y + vy;

        ctx.strokeStyle = p.c + '0.30)';
        ctx.lineWidth = p.w;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(nx, ny);
        ctx.stroke();

        p.x = nx; p.y = ny; p.life--;
        if (p.life < 0 || p.x > W + 20 || p.y < -20 || p.y > H + 20) {
          p.x = -10 + Math.random() * 40;
          p.y = Math.random() * H;
          p.life = 160 + Math.random() * 220;
        }
      }
      raf = requestAnimationFrame(frame);
    }

    let raf = null;
    function start() {
      resize(); seed();
      if (raf) cancelAnimationFrame(raf);
      if (reduceMotion) { for (let k = 0; k < 90; k++) stepStatic(); return; }
      frame();
    }
    function stepStatic() {
      tt += 0.016;
      for (let i = 0; i < parts.length; i++) {
        const p = parts[i];
        const ang = field(p.x, p.y, tt);
        const nx = p.x + Math.cos(ang) * 1.5 + 0.85, ny = p.y + Math.sin(ang) * 1.5;
        ctx.strokeStyle = p.c + '0.22)';
        ctx.lineWidth = p.w;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(nx, ny); ctx.stroke();
        p.x = nx; p.y = ny;
        if (p.x > W + 20 || p.y < -20 || p.y > H + 20) { p.x = Math.random() * 30; p.y = Math.random() * H; }
      }
    }

    start();
    let rz;
    window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(start, 220); });

    // pause when scrolled out of view
    if ('IntersectionObserver' in window && !reduceMotion) {
      new IntersectionObserver((es) => {
        es.forEach(e => {
          if (e.isIntersecting) { if (!raf) frame(); }
          else { if (raf) { cancelAnimationFrame(raf); raf = null; } }
        });
      }, { threshold: 0 }).observe(cv.parentElement);
    }
  })();

  /* ───────────────────────── 2. TOC scrollspy ───────────────────────── */
  (function scrollspy() {
    const links = $$('#toc nav a');
    if (!links.length) return;
    const map = new Map();
    links.forEach(a => {
      const el = document.getElementById(a.getAttribute('href').slice(1));
      if (el) map.set(el, a);
    });
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          links.forEach(l => l.classList.remove('is-active'));
          const a = map.get(e.target);
          if (a) a.classList.add('is-active');
        }
      });
    }, { rootMargin: '-10% 0px -72% 0px', threshold: 0 });
    map.forEach((_, el) => io.observe(el));
  })();

  /* ───────────────────────── 3. paradigm filter ───────────────────────── */
  (function paradigms() {
    const cards = $$('.para-card');
    const table = $('#prop-table');
    const cap   = $('#para-caption');
    if (!cards.length || !table) return;

    const notes = {
      discrete: 'Fully discrete models get a shared objective and a shared sampler for free — both modalities are categorical. The price is the visual tokenizer: detail it discards is gone before the backbone ever sees it.',
      hybrid:   'Hybrids keep continuous visual states, so fidelity survives. But text is predicted autoregressively while images are denoised, so one backbone has to serve two objectives and two sampling procedures.',
      flow:     'Multimodal Flow keeps continuous states for both modalities and generates them with one vector field. Modality-specific structure lives in the frozen encoders and decoders, not in the generative process.'
    };
    const dflt = cap.innerHTML;

    function select(p) {
      cards.forEach(c => {
        const on = c.dataset.p === p;
        c.classList.toggle('is-sel', on);
        c.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      table.classList.toggle('is-filtered', !!p);
      $$('[data-col]', table).forEach(el => el.classList.toggle('col-on', el.dataset.col === p));
      cap.innerHTML = p ? notes[p] : dflt;
    }

    let cur = null;
    cards.forEach(c => c.addEventListener('click', () => {
      cur = (cur === c.dataset.p) ? null : c.dataset.p;
      select(cur);
    }));
  })();

  /* ───────────────────────── 4. chunk construction ───────────────────────── */
  (function chunkDemo() {
    const root = $('#chunk-demo');
    if (!root) return;

    const SENT = ('A lifelike astronaut floats peacefully above the softly glowing horizon of a ' +
                  'distant planet while the nebula burns quietly behind it').split(' ');
    const B = 8;

    // raw text with per-chunk tinting
    const raw = $('#raw-text');
    raw.innerHTML = SENT.map((w, i) =>
      `<span class="tok" data-ch="${Math.floor(i / B)}">${w}</span>`).join(' ');

    // chunk cards
    const row = $('#text-chunks');
    const nCh = Math.ceil(SENT.length / B);
    let html = '';
    for (let c = 0; c < nCh; c++) {
      const n = Math.min(B, SENT.length - c * B);
      let toks = '';
      for (let j = 0; j < B; j++) toks += `<i class="${j < n ? '' : 'empty'}"></i>`;
      html += `<div class="tchunk" data-ch="${c}">
                 <div class="tchunk-head">c<sub>${c + 1}</sub><sup>ℓ</sup></div>
                 <div class="tchunk-toks">${toks}</div>
               </div>`;
    }
    row.innerHTML = html;

    // hover linking between raw tokens and chunk cards
    function hl(ch) {
      $$('.tok', raw).forEach(t => t.classList.toggle('hl', ch !== null && t.dataset.ch === String(ch)));
      $$('.tchunk', row).forEach(t => {
        t.style.transform = (ch !== null && t.dataset.ch === String(ch)) ? 'translateY(-2px)' : '';
        t.style.boxShadow = (ch !== null && t.dataset.ch === String(ch)) ? '0 2px 8px rgba(16,24,40,.12)' : '';
      });
    }
    $$('.tchunk', row).forEach(t => {
      t.addEventListener('mouseenter', () => hl(t.dataset.ch));
      t.addEventListener('mouseleave', () => hl(null));
    });
    $$('.tok', raw).forEach(t => {
      t.addEventListener('mouseenter', () => hl(t.dataset.ch));
      t.addEventListener('mouseleave', () => hl(null));
    });
    // cycle a highlight on load so the link is discoverable
    if (!reduceMotion) {
      let k = 0;
      const iv = setInterval(() => { hl(k % nCh); k++; if (k > nCh) { clearInterval(iv); hl(null); } }, 620);
    }

    // 16x16 patch grid, tinted from the source image
    const grid = $('#grid16');
    if (grid) {
      let g = '';
      for (let i = 0; i < 256; i++) g += '<i></i>';
      grid.innerHTML = g;
      const img = new Image();
      img.onload = function () {
        const c = document.createElement('canvas');
        c.width = 16; c.height = 16;
        const cx = c.getContext('2d');
        cx.drawImage(img, 0, 0, 16, 16);
        let d;
        try { d = cx.getImageData(0, 0, 16, 16).data; } catch (e) { return; }
        const cells = grid.children;
        for (let i = 0; i < 256; i++) {
          const o = i * 4;
          cells[i].style.background = `rgb(${d[o]},${d[o + 1]},${d[o + 2]})`;
        }
      };
      img.src = 'assets/nebula.png';
    }

    // text / image switch
    $$('.seg-btn', root).forEach(b => b.addEventListener('click', () => {
      $$('.seg-btn', root).forEach(x => x.classList.toggle('is-on', x === b));
      $$('.chunk-pane', root).forEach(p => p.classList.toggle('is-on', p.dataset.pane === b.dataset.mode));
    }));
  })();

  /* ───────────────────────── 5. chunk-causal mask ───────────────────────── */
  (function maskDemo() {
    const grid = $('#mask-grid');
    if (!grid) return;
    const colsEl = $('#mask-cols'), rowsEl = $('#mask-rows'), cap = $('#mask-caption');
    const K = 7;
    const MOD = ['text', 'text', 'text', 'vision', 'text', 'text', 'vision'];
    const WHO = ['text A', 'text A', 'text A', 'image', 'text B', 'text B', 'image'];
    const dflt = cap.innerHTML;

    let h = '';
    for (let i = 1; i <= K; i++) h += `<span>c<sub>${i}</sub></span>`;
    colsEl.innerHTML = h;
    rowsEl.innerHTML = h;

    let g = '';
    for (let r = 0; r < K; r++) {
      for (let c = 0; c < K; c++) {
        const cls = c < r ? 'past' : (c === r ? 'cur' : 'fut');
        g += `<div class="mask-cell ${cls}" data-r="${r}"></div>`;
      }
    }
    grid.innerHTML = g;

    function focus(r) {
      grid.classList.toggle('is-focus', r !== null);
      $$('.mask-cell', grid).forEach(c => c.classList.toggle('row-on', r !== null && +c.dataset.r === r));
      $$('span', rowsEl).forEach((s, i) => s.classList.toggle('row-on', r === i));

      if (r === null) { cap.innerHTML = dflt; return; }
      const name = `c<sub>${r + 1}</sub>`;
      const mod  = MOD[r] === 'text' ? 'a text chunk' : 'a visual chunk';
      if (r === 0) {
        cap.innerHTML = `<strong>Target ${name}</strong> is ${mod} with no context at all — it is
          generated unconditionally. It sees only its own noisy positions, bidirectionally;
          c<sub>2</sub>–c<sub>7</sub> are masked.`;
      } else {
        cap.innerHTML = `<strong>Target ${name}</strong> is ${mod} from <em>${WHO[r]}</em>. It attends to the
          <strong>clean</strong> states of c<sub>1</sub>–c<sub>${r}</sub> — across both modalities — plus its own
          noisy positions. ${r === K - 1 ? 'Nothing follows it.' :
          `c<sub>${r + 2}</sub>–c<sub>${K}</sub> are masked.`}
          Note it never sees its own clean counterpart.`;
      }
    }

    $$('.mask-cell', grid).forEach(c => {
      c.addEventListener('mouseenter', () => focus(+c.dataset.r));
      c.addEventListener('click',      () => focus(+c.dataset.r));
    });
    $$('span', rowsEl).forEach((s, i) => {
      s.addEventListener('mouseenter', () => focus(i));
      s.addEventListener('click',      () => focus(i));
    });
    grid.addEventListener('mouseleave', () => focus(null));
    rowsEl.addEventListener('mouseleave', () => focus(null));
  })();

  /* ───────────────────────── 6. flow-matching t demo ───────────────────────── */
  (function flowDemo() {
    const slider = $('#t-slider');
    if (!slider) return;
    const out = $('#t-out'), txtEl = $('#flow-text'), img = $('#flow-img'), cv = $('#flow-noise');
    const playBtn = $('#play-btn');

    const WORDS = ['A', 'lifelike', 'astronaut', 'floats', 'peacefully', 'above', 'the', 'horizon'];
    const GLYPH = '▚▞▓▒░▘▗◣◥╳┼≋∿⌁⋰⋱';

    // deterministic per-token / per-char randomness so the animation is stable
    let s = 20260101;
    const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

    const toks = WORDS.map(w => ({
      w,
      thr: 0.18 + rnd() * 0.62,                       // when this token resolves
      cv: w.split('').map(() => rnd()),               // per-char reveal order
      g:  w.split('').map(() => GLYPH[(rnd() * GLYPH.length) | 0])
    }));
    txtEl.innerHTML = toks.map(() => '<span class="ftok"></span>').join('');
    const spans = $$('.ftok', txtEl);

    // noise canvas
    const NW = 72;
    const nctx = cv.getContext('2d');
    cv.width = NW; cv.height = NW;
    function noise() {
      const d = nctx.createImageData(NW, NW);
      for (let i = 0; i < NW * NW; i++) {
        const o = i * 4;
        const v = 120 + (Math.random() - 0.5) * 240;
        d.data[o]     = clamp(v + (Math.random() - 0.5) * 60, 0, 255);
        d.data[o + 1] = clamp(v + (Math.random() - 0.5) * 60, 0, 255);
        d.data[o + 2] = clamp(v + (Math.random() - 0.5) * 60, 0, 255);
        d.data[o + 3] = 255;
      }
      nctx.putImageData(d, 0, 0);
    }
    noise();

    let lastNoise = 0;
    function render(t) {
      out.textContent = t.toFixed(2);

      // text: tokens resolve out of noise at their own thresholds
      toks.forEach((tk, i) => {
        const rev = clamp((t - tk.thr) / 0.14 + 0.5, 0, 1);
        const sp = spans[i];
        if (rev >= 1) {
          sp.textContent = tk.w;
          sp.classList.add('on');
          sp.style.filter = '';
        } else {
          sp.textContent = tk.w.split('').map((ch, j) => (tk.cv[j] < rev ? ch : tk.g[j])).join('');
          sp.classList.toggle('on', rev > 0.55);
          sp.style.filter = `blur(${(1 - rev) * 1.1}px)`;
        }
      });

      // image: noise mixed out, blur and saturation recovering
      const u = 1 - t;
      cv.style.opacity = String(Math.pow(u, 0.75));
      img.style.filter = `blur(${u * 7}px) saturate(${clamp(0.25 + t * 0.95, 0, 1.2)}) contrast(${0.8 + t * 0.25})`;

      const now = performance.now();
      if (u > 0.02 && now - lastNoise > 70) { noise(); lastNoise = now; }
    }

    let raf = null;
    function onInput() {
      if (raf) return;
      raf = requestAnimationFrame(() => { raf = null; render(slider.value / 1000); });
    }
    slider.addEventListener('input', onInput);

    // play
    let playing = null;
    function stop() {
      if (playing) cancelAnimationFrame(playing);
      playing = null;
      playBtn.textContent = '▶ Animate';
    }
    playBtn.addEventListener('click', () => {
      if (playing) { stop(); return; }
      playBtn.textContent = '■ Stop';
      let v = 0;
      const step = () => {
        v += 0.0052;
        if (v >= 1) { v = 1; slider.value = 1000; render(1); stop(); return; }
        slider.value = Math.round(v * 1000);
        render(v);
        playing = requestAnimationFrame(step);
      };
      playing = requestAnimationFrame(step);
    });

    render(0);

    // auto-play once when scrolled into view
    if ('IntersectionObserver' in window && !reduceMotion) {
      let fired = false;
      new IntersectionObserver((es) => {
        es.forEach(e => {
          if (e.isIntersecting && !fired) { fired = true; setTimeout(() => playBtn.click(), 380); }
        });
      }, { threshold: 0.45 }).observe($('#flow-demo'));
    }
  })();

  /* ───────────────────────── 7. tasks as chunk orders ───────────────────────── */
  (function taskDemo() {
    const root = $('#task-demo');
    if (!root) return;
    const strip = $('#task-strip'), desc = $('#task-desc');

    const T = (role) => ({ m: 'text', role });
    const V = (role) => ({ m: 'vision', role });

    const TASKS = {
      textonly: {
        seq: [T('tgt'), T('tgt'), T('tgt'), T('tgt'), T('tgt')],
        d: '<strong>Text-only language modeling.</strong> Consecutive 8-token blocks; every chunk is a target conditioned on its clean prefix, so the first one is generated unconditionally. This is an ordinary language model — expressed as a flow over embeddings rather than a softmax over a vocabulary.'
      },
      imageonly: {
        seq: [V('tgt')],
        d: '<strong>Image-only modeling.</strong> A single visual target with no conditioning chunk at all. The degenerate case of the same formulation: an unconditional image generator.'
      },
      t2i: {
        seq: [T('ctx'), T('ctx'), T('ctx'), V('tgt')],
        d: '<strong>Text → image.</strong> Caption blocks form the clean prefix; the image is the flow target. All 256 visual positions are produced jointly, inside one chunk.'
      },
      i2t: {
        seq: [V('ctx'), T('tgt'), T('tgt'), T('tgt')],
        d: '<strong>Image → text.</strong> The same pair, the other way round. The image is clean context and the caption blocks become sequential targets — which is why pretraining learns <em>bidirectional</em> conditionals, not just one direction.'
      },
      vqa: {
        seq: [V('ctx'), T('ctx'), T('ctx'), T('tgt'), T('tgt')],
        d: '<strong>Visual question answering.</strong> Image and question chunks are clean context; answer blocks are the targets. Compared with image→text pretraining, only the data and the sequence shape changed — not the interface, the factorization, or the loss.'
      },
      t2ift: {
        seq: [T('ctx'), T('ctx'), T('ctx'), T('ctx'), V('tgt')],
        d: '<strong>Text-to-image finetuning.</strong> Prompt chunks as prefix, image as target. Structurally identical to the pretraining task above it — the backbone is simply initialized from mixed pretraining and optimized on prompt-image data.'
      }
    };

    function render(key) {
      const t = TASKS[key];
      let h = '';
      t.seq.forEach((c, i) => {
        if (i) h += '<span class="tarrow">→</span>';
        h += `<div class="tcell ${c.role} ${c.m}" style="animation-delay:${i * 55}ms">
                <span class="tcell-name">c<sub>${i + 1}</sub></span>
                <span class="tcell-role">${c.role === 'ctx' ? 'context' : 'target'}</span>
              </div>`;
      });
      strip.innerHTML = h;
      desc.innerHTML = t.d;
    }

    $$('.task-btn', root).forEach(b => b.addEventListener('click', () => {
      $$('.task-btn', root).forEach(x => x.classList.toggle('is-on', x === b));
      render(b.dataset.task);
    }));
    render('textonly');
  })();

  /* ───────────────────────── 8. training vs inference ───────────────────────── */
  (function phaseDemo() {
    const root = $('#phase-demo');
    if (!root) return;
    const strip = $('#phase-strip'), desc = $('#phase-desc');
    const MOD = ['text', 'text', 'text', 'vision', 'text', 'text', 'vision'];
    let timer = null, raf = null;

    function build() {
      strip.innerHTML = MOD.map((m, i) =>
        `<div class="pcell ${m}" data-i="${i}">
           <span class="pcell-name">c<sub>${i + 1}</sub></span>
           <span class="pcell-bar"><i style="width:0%"></i></span>
         </div>`).join('');
      return $$('.pcell', strip);
    }

    function clear() {
      if (timer) { clearInterval(timer); timer = null; }
      if (raf) { cancelAnimationFrame(raf); raf = null; }
    }

    function train() {
      clear();
      const cells = build();
      cells.forEach(c => c.classList.add('active'));
      // each target gets its own independently sampled timestep → its own fill rate
      const rate = cells.map(() => 0.004 + Math.random() * 0.012);
      const prog = cells.map(() => 0);
      desc.innerHTML = '<strong>Training.</strong> All seven chunks are targets at once. Each gets its own ' +
        'independently sampled timestep (shifted logit-normal, <span class="m">α = 8</span> for images, ' +
        '<span class="m">α = 6</span> for text) and its own noisy view, and every one contributes a Flow Matching ' +
        'loss in the <em>same forward pass</em>. The bars fill at different rates because the timesteps differ.';
      if (reduceMotion) { cells.forEach(c => { c.classList.add('done'); }); return; }
      const step = () => {
        let alive = false;
        cells.forEach((c, i) => {
          if (prog[i] < 1) { prog[i] = Math.min(1, prog[i] + rate[i]); alive = true; }
          c.querySelector('.pcell-bar i').style.width = (prog[i] * 100) + '%';
          if (prog[i] >= 1) c.classList.add('done');
        });
        if (alive) raf = requestAnimationFrame(step);
        else { raf = null; setTimeout(() => { if (root.dataset.phase === 'train') train(); }, 1400); }
      };
      raf = requestAnimationFrame(step);
    }

    function infer() {
      clear();
      const cells = build();
      cells.forEach(c => c.classList.add('pending'));
      desc.innerHTML = '<strong>Inference.</strong> Chunks are produced one at a time, each by integrating the ' +
        'vector field from <span class="m">t = 0</span> to <span class="m">t = 1</span> conditioned on everything ' +
        'already completed. A finished chunk is appended to the KV cache before the next one starts — so the clean ' +
        'prefix is encoded once, not re-encoded per sampling step.';
      if (reduceMotion) { cells.forEach(c => { c.classList.remove('pending'); c.classList.add('done'); }); return; }
      let i = 0, p = 0;
      const step = () => {
        if (i >= cells.length) {
          raf = null;
          setTimeout(() => { if (root.dataset.phase === 'infer') infer(); }, 1500);
          return;
        }
        const c = cells[i];
        c.classList.remove('pending');
        c.classList.add('active');
        p += 0.022;
        c.querySelector('.pcell-bar i').style.width = Math.min(100, p * 100) + '%';
        if (p >= 1) { c.classList.remove('active'); c.classList.add('done'); i++; p = 0; }
        raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    }

    root.dataset.phase = 'train';
    $$('.seg-btn', root).forEach(b => b.addEventListener('click', () => {
      $$('.seg-btn', root).forEach(x => x.classList.toggle('is-on', x === b));
      root.dataset.phase = b.dataset.phase;
      (b.dataset.phase === 'train' ? train : infer)();
    }));
    train();
  })();

  /* ───────────────────── 9. modality palette + sequence composer ───────────────────── */

  // One shared registry: a chunk type is an encoder, a chunking rule and a decoder.
  // `trained: false` means the formulation admits it but MF-1 never saw it.
  const CHUNKS = {
    text: {
      name: 'Text', short: 'T', color: '#3f72b0', soft: '#dde8f5', line: '#a8c4e4', deep: '#24497a',
      trained: true,
      rule: 'a contiguous run of <b>8 tokens</b> — short enough to keep word order meaningful, long enough to be worth one pass',
      codec: 'T5-small → 8 × 512 → text decoder'
    },
    image: {
      name: 'Image', short: 'I', color: '#5d9b4b', soft: '#ddefd4', line: '#a9d396', deep: '#3d6b30',
      trained: true,
      rule: 'the <b>entire 16 × 16 grid</b> — an image has no natural reading order, so all of it is generated at once',
      codec: 'SigLIP2-so400m → 256 × 1152 → RAE decoder'
    },
    video: {
      name: 'Video', short: 'V', color: '#4a8fa8', soft: '#d9ecf2', line: '#9ccddc', deep: '#2e6579',
      trained: false,
      rule: 'one <b>short clip or single frame</b> — the grid is spatial, the chunk boundary is temporal',
      codec: 'frozen video / per-frame encoder → F × 256 × d'
    },
    audio: {
      name: 'Audio', short: 'A', color: '#b07a3f', soft: '#f5e8d6', line: '#e0c49a', deep: '#7a5223',
      trained: false,
      rule: 'a <b>fixed-length window</b> of latent frames — the same move as a text block, on a different axis',
      codec: 'frozen audio encoder → W × d → vocoder'
    },
    depth: {
      name: 'Depth / mask', short: 'D', color: '#7a8fa0', soft: '#e6ecf0', line: '#bccbd6', deep: '#4a5c6b',
      trained: false,
      rule: 'the <b>whole map</b>, on the image grid — a second view of the same scene, aligned position by position',
      codec: 'frozen dense encoder → 256 × d'
    },
    action: {
      name: 'Action', short: 'Ac', color: '#9c5c8f', soft: '#f0e0ed', line: '#d6b0cc', deep: '#6b3861',
      trained: false,
      rule: 'a <b>short control horizon</b> — the steps ahead that have to be decided together, not one at a time',
      codec: 'normalizer → H × d → controller'
    }
  };

  (function modalityCards() {
    const host = $('#modalities');
    if (!host) return;
    host.innerHTML = Object.keys(CHUNKS).map(k => {
      const c = CHUNKS[k];
      return `<div class="mod-card ${c.trained ? '' : 'spec'}">
        <div class="mod-head">
          <span class="mod-ico" style="background:${c.color}">${c.short}</span>
          <span class="mod-name">${c.name}</span>
        </div>
        <p class="mod-rule"><span style="color:${c.deep};font-weight:600">Natural chunk:</span> ${c.rule}</p>
        <div class="mod-codec">${c.codec}</div>
      </div>`;
    }).join('');
  })();

  (function composer() {
    const root = $('#compose-demo');
    if (!root) return;
    const paletteEl = $('#comp-palette'), stripEl = $('#comp-strip'), maskEl = $('#comp-mask');
    const taskEl = $('#comp-task'), descEl = $('#comp-desc'), badgeEl = $('#comp-badge');
    const presetEl = $('#comp-presets');

    // sequence: [{type, role}] with role 'ctx' | 'tgt'
    let seq = [
      { type: 'image',  role: 'ctx' },
      { type: 'text',   role: 'ctx' },
      { type: 'action', role: 'tgt' }
    ];

    const PRESETS = {
      'Visual QA':            [['image','ctx'], ['text','ctx'], ['text','tgt']],
      'Image editing':        [['image','ctx'], ['text','ctx'], ['image','tgt']],
      'Interleaved story':    [['text','tgt'], ['image','tgt'], ['text','tgt'], ['image','tgt']],
      'Video continuation':   [['video','ctx'], ['video','ctx'], ['video','tgt']],
      'Narrated video':       [['video','ctx'], ['video','ctx'], ['audio','tgt'], ['text','tgt']],
      'Driving policy':       [['image','ctx'], ['image','ctx'], ['text','ctx'], ['action','tgt']],
      'Speech recognition':   [['audio','ctx'], ['audio','ctx'], ['text','tgt']],
      'Depth estimation':     [['image','ctx'], ['depth','tgt']]
    };

    // recognised (context → target) shapes, keyed by de-duplicated type signature
    const NAMED = {
      'text>text':              ['Language modeling', 'Ordinary next-block language modeling, as a flow over embeddings.'],
      '>text':                  ['Text-only language modeling', 'Every block is a target conditioned on its clean prefix, so the first one is generated unconditionally.'],
      '>image':                 ['Image-only modeling', 'A single visual target with no conditioning chunk — an unconditional image generator.'],
      'text>image':             ['Text-to-image generation', 'Caption blocks as clean prefix, the image as the flow target.'],
      'image>text':             ['Image captioning', 'The image is clean context; caption blocks become sequential targets.'],
      'image,text>text':        ['Visual question answering', 'Image and question as context, answer blocks as targets.'],
      'image,text>image':       ['Instruction-guided image editing', 'A source image plus an instruction, generating a new image.'],
      'image>image':            ['Image-to-image translation', 'One visual chunk conditioning another.'],
      'image>depth':            ['Monocular depth estimation', 'Same spatial grid, different output space — the chunk shape does not even change.'],
      'video>video':            ['Video continuation', 'Earlier clips as context, the next clip as target.'],
      'video>audio':            ['Video-to-audio', 'Generating a soundtrack conditioned on what is on screen.'],
      'video>audio,text':       ['Video narration', 'One pass producing both a soundtrack and a transcript.'],
      'video>text':             ['Video captioning', 'Temporal context, textual target.'],
      'audio>text':             ['Speech recognition', 'Audio windows as context, text blocks as targets.'],
      'text>audio':             ['Speech synthesis', 'Text as prefix, audio windows generated in order.'],
      'image>action':           ['Visuomotor policy', 'Observation in, control horizon out.'],
      'image,text>action':      ['Language-conditioned control', 'An instruction and what the camera sees, producing an action chunk.'],
      'video,text>action':      ['Instruction-following from video', 'Temporal observation plus an instruction, producing control.'],
      'text>text,image':        ['Illustrated generation', 'Text prompt producing interleaved prose and imagery.'],
      '>text,image':            ['Interleaved document generation', 'Alternating text and image targets, each conditioned on everything before it.']
    };

    // The five chunk configurations MF-1 was actually trained on (pretraining mixture
    // plus downstream finetuning). Everything else is an arrangement the formulation
    // allows but that we never optimized.
    const TRAINED_TASKS = {
      '>text': 1,            // text-only language modeling
      '>image': 1,           // image-only modeling
      'text>image': 1,       // text-to-image
      'image>text': 1,       // image-to-text / captioning
      'image,text>text': 1   // visual question answering
    };

    function uniq(a) { return a.filter((v, i) => a.indexOf(v) === i); }

    function describe() {
      const ctx = seq.filter(c => c.role === 'ctx');
      const tgt = seq.filter(c => c.role === 'tgt');
      const ctxT = uniq(ctx.map(c => c.type));
      const tgtT = uniq(tgt.map(c => c.type));
      const sig = ctxT.join(',') + '>' + tgtT.join(',');
      const named = NAMED[sig];

      const label = ts => ts.map(t => CHUNKS[t].name.toLowerCase()).join(' + ');
      let title, desc;

      if (!seq.length) {
        return { title: 'Nothing yet', desc: 'Add chunks from the palette above to build a sequence.', state: 'empty' };
      }
      if (!tgt.length) {
        return {
          title: 'No target',
          desc: 'Every sequence needs at least one target chunk — that is what the Flow Matching loss is computed on. Click a chunk to flip it to <b>target</b>.',
          state: 'empty'
        };
      }

      if (named) { title = named[0]; desc = named[1]; }
      else {
        title = ctxT.length
          ? label(ctxT).replace(/^./, s => s.toUpperCase()) + ' → ' + label(tgtT)
          : 'Unconditional ' + label(tgtT) + ' generation';
        desc = ctxT.length
          ? `Given <b>${label(ctxT)}</b> as clean context, generate <b>${label(tgtT)}</b> — one chunk at a time, in the order you placed them.`
          : `Generate <b>${label(tgtT)}</b> from noise, each chunk conditioned on the ones already completed.`;
      }

      const untrained = uniq(seq.map(c => c.type)).filter(t => !CHUNKS[t].trained);
      let state;
      if (untrained.length) state = 'spec';            // a modality MF-1 never saw
      else if (TRAINED_TASKS[sig]) state = 'ok';       // an objective we actually optimized
      else state = 'warn';                             // trained modalities, untrained arrangement
      return { title, desc, state, untrained };
    }

    function render() {
      // strip
      if (!seq.length) {
        stripEl.innerHTML = '<div class="comp-empty">Empty sequence — add a chunk from the palette above.</div>';
      } else {
        stripEl.innerHTML = seq.map((c, i) => {
          const k = CHUNKS[c.type];
          const style = c.role === 'tgt'
            ? `background:repeating-linear-gradient(45deg,${k.soft},${k.soft} 4px,#ffffff 4px,#ffffff 8px);border:1.5px dashed var(--purple);color:${k.deep}`
            : `background:${k.soft};border:1px solid ${k.line};color:${k.deep}`;
          return (i ? '<span class="tarrow">→</span>' : '') +
            `<div class="ccell ${c.role}" data-i="${i}" style="${style};animation-delay:${i * 45}ms" title="click to flip context / target">
               <span class="ccell-x" data-rm="${i}">×</span>
               <span class="ccell-i">c<sub>${i + 1}</sub></span>
               <span class="ccell-n">${k.name}</span>
               <span class="ccell-r">${c.role === 'ctx' ? 'context' : 'target'}</span>
             </div>`;
        }).join('');
      }

      // mask, derived from the sequence the reader built
      const K = seq.length;
      if (!K) {
        maskEl.style.gridTemplateColumns = '';
        maskEl.innerHTML = '<span class="comp-mask-none">—</span>';
      } else {
        maskEl.style.gridTemplateColumns = `repeat(${K}, 24px)`;
        let g = '';
        for (let r = 0; r < K; r++) {
          for (let c = 0; c < K; c++) {
            // a context chunk is never a target row, so its row is simply its prefix
            const cls = c < r ? 'past' : (c === r ? (seq[r].role === 'tgt' ? 'cur' : 'past') : 'fut');
            g += `<div class="mask-cell ${cls}"></div>`;
          }
        }
        maskEl.innerHTML = g;
      }

      // reading
      const d = describe();
      taskEl.textContent = d.title;
      descEl.innerHTML = d.desc;
      if (d.state === 'empty') {
        badgeEl.className = 'comp-badge spec';
        badgeEl.textContent = '';
        badgeEl.style.display = 'none';
      } else {
        badgeEl.style.display = '';
        if (d.state === 'ok') {
          badgeEl.className = 'comp-badge ok';
          badgeEl.innerHTML = '✓&nbsp; an objective MF-1 was actually trained on';
        } else if (d.state === 'warn') {
          badgeEl.className = 'comp-badge warn';
          badgeEl.innerHTML = 'familiar modalities · <b>this arrangement was never trained</b>';
        } else {
          badgeEl.className = 'comp-badge spec';
          badgeEl.innerHTML = 'new modality · admitted by the formulation, <b>not trained</b>';
        }
      }
    }

    // palette
    paletteEl.innerHTML = Object.keys(CHUNKS).map(k => {
      const c = CHUNKS[k];
      return `<button class="pal-btn ${c.trained ? '' : 'spec'}" data-add="${k}">
                <span class="pal-plus">+</span>
                <span class="pal-ico" style="background:${c.color}">${c.short}</span>${c.name}
              </button>`;
    }).join('');

    // presets
    Object.keys(PRESETS).forEach(name => {
      const b = document.createElement('button');
      b.className = 'cp-btn';
      b.textContent = name;
      b.addEventListener('click', () => {
        seq = PRESETS[name].map(p => ({ type: p[0], role: p[1] }));
        render();
      });
      presetEl.appendChild(b);
    });

    paletteEl.addEventListener('click', e => {
      const b = e.target.closest('[data-add]');
      if (!b || seq.length >= 9) return;
      seq.push({ type: b.dataset.add, role: 'tgt' });
      render();
    });

    stripEl.addEventListener('click', e => {
      const rm = e.target.closest('[data-rm]');
      if (rm) { seq.splice(+rm.dataset.rm, 1); render(); return; }
      const cell = e.target.closest('[data-i]');
      if (!cell) return;
      const c = seq[+cell.dataset.i];
      c.role = c.role === 'ctx' ? 'tgt' : 'ctx';
      render();
    });

    $('#seq-clear').addEventListener('click', () => { seq = []; render(); });

    render();
  })();

  /* ───────────────────────── 10. copy bibtex ───────────────────────── */
  (function copyBib() {
    const btn = $('#copy-bib');
    if (!btn) return;
    btn.addEventListener('click', () => {
      const code = $('.bib code').innerText;
      const done = () => { btn.textContent = 'Copied ✓'; setTimeout(() => (btn.textContent = 'Copy'), 1600); };
      if (navigator.clipboard) navigator.clipboard.writeText(code).then(done).catch(done);
      else {
        const ta = document.createElement('textarea');
        ta.value = code; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); } catch (e) {}
        document.body.removeChild(ta); done();
      }
    });
  })();

})();

/* MF-1 demo: the "resolve" reveal for text answers.
   Runs once on page load. Everything is optional polish: any failure leaves the page
   fully usable. Gradio mounts its DOM late, so a MutationObserver finds the elements. */
(() => {
  try {
    if (window.__mfDemoInit) return;
    window.__mfDemoInit = true;

    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* ── text reveal: glyphs resolve into the final answer, left to right ── */
    const GLYPHS = '▚▞▓▒░▘▗◣◥╳┼≋∿⌁';
    function reveal(el) {
      const full = el.textContent;
      if (reduce || !full || full.length > 1500) return;
      const chars = Array.from(full);
      const order = chars.map((_, i) => i / chars.length * 0.6 + Math.random() * 0.4);
      const t0 = performance.now(), DUR = Math.min(1400, 350 + chars.length * 6);
      function tick(now) {
        if (!el.isConnected) return;
        const p = Math.min((now - t0) / DUR, 1);
        el.textContent = chars.map((ch, i) =>
          (/\s/.test(ch) || order[i] < p) ? ch : GLYPHS[(Math.random() * GLYPHS.length) | 0]).join('');
        if (p < 1) requestAnimationFrame(tick); else el.textContent = full;
      }
      requestAnimationFrame(tick);
    }

    let pending = false;
    function scan() {
      pending = false;
      document.querySelectorAll('.mf-reveal:not([data-done])').forEach((el) => { el.dataset.done = '1'; reveal(el); });
    }
    new MutationObserver(() => { if (!pending) { pending = true; requestAnimationFrame(scan); } })
      .observe(document.body, { childList: true, subtree: true });
    scan();
  } catch (err) {
    console.warn('mf-demo animations disabled:', err);
  }
})();

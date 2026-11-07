import React, { useEffect, useState } from 'react';
import { ArrowUpRight } from 'lucide-react';

const mono = { fontFamily: 'var(--font-mono)' };
const loadImg = (src) => new Promise((res) => { const i = new Image(); i.onload = () => res(i); i.src = src; });

// Inferno-style colour stops (dark -> bright) for the illumination map, as in the reference figure.
const STOPS = [[0, 0, 4], [66, 10, 104], [147, 38, 103], [221, 81, 58], [252, 165, 10], [252, 255, 164]];
const colour = (t) => {
  const p = Math.min(0.9999, Math.max(0, t)) * (STOPS.length - 1), i = Math.floor(p), f = p - i;
  return STOPS[i].map((v, k) => Math.round(v + (STOPS[i + 1][k] - v) * f));
};

function gaussianBlur(src, w, h, sigma) {
  const r = Math.ceil(sigma * 3), k = [];
  let sum = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k.push(v); sum += v; }
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = -r; j <= r; j++) s += k[j + r] * src[y * w + Math.min(w - 1, Math.max(0, x + j))];
    tmp[y * w + x] = s / sum;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = -r; j <= r; j++) s += k[j + r] * tmp[Math.min(h - 1, Math.max(0, y + j)) * w + x];
    out[y * w + x] = s / sum;
  }
  return out;
}

/* Diagnostic visualisations computed in the browser from the user's real uploaded image.
   Classical image processing only; these are not learned model outputs and the pipeline does not store them. */
async function computeMaps(src) {
  const img = await loadImg(src);
  const w = img.naturalWidth, h = img.naturalHeight, n = w * h;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const px = ctx.getImageData(0, 0, w, h).data;
  const lum = (d, i) => 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2];
  const g = new Float32Array(n);
  for (let i = 0; i < n; i++) g[i] = lum(px, i);

  // 2. Estimated illumination: Gaussian low-pass of luminance (on a 128 px-wide copy), smoothly upscaled
  const dw = 128, dh = Math.max(2, Math.round((128 * h) / w));
  const small = document.createElement('canvas');
  small.width = dw; small.height = dh;
  const sctx = small.getContext('2d', { willReadFrequently: true });
  sctx.drawImage(img, 0, 0, dw, dh);
  const sd = sctx.getImageData(0, 0, dw, dh);
  const sl = new Float32Array(dw * dh);
  for (let i = 0; i < dw * dh; i++) sl[i] = lum(sd.data, i);
  const blurred = gaussianBlur(sl, dw, dh, 5);
  let lo = Infinity, hi = -Infinity;
  blurred.forEach((v) => { if (v < lo) lo = v; if (v > hi) hi = v; });
  for (let i = 0; i < dw * dh; i++) sd.data.set([...colour((blurred[i] - lo) / Math.max(1, hi - lo)), 255], 4 * i);
  sctx.putImageData(sd, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, w, h);
  const illum = c.toDataURL('image/jpeg', 0.9);

  // 3. Canny: gaussian blur -> Sobel -> non-maximum suppression -> double threshold -> hysteresis
  const b = gaussianBlur(g, w, h, 1.4);
  const mag = new Float32Array(n), dir = new Uint8Array(n);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const gx = -b[i - w - 1] - 2 * b[i - 1] - b[i + w - 1] + b[i - w + 1] + 2 * b[i + 1] + b[i + w + 1];
    const gy = -b[i - w - 1] - 2 * b[i - w] - b[i - w + 1] + b[i + w - 1] + 2 * b[i + w] + b[i + w + 1];
    mag[i] = Math.hypot(gx, gy);
    const a = (Math.atan2(gy, gx) * 180) / Math.PI, ang = a < 0 ? a + 180 : a;
    dir[i] = ang < 22.5 || ang >= 157.5 ? 0 : ang < 67.5 ? 1 : ang < 112.5 ? 2 : 3;
  }
  const nms = new Float32Array(n), vals = [];
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x, m = mag[i];
    if (m === 0) continue;
    let p, q;
    if (dir[i] === 0) { p = mag[i - 1]; q = mag[i + 1]; }
    else if (dir[i] === 1) { p = mag[i - w + 1]; q = mag[i + w - 1]; }
    else if (dir[i] === 2) { p = mag[i - w]; q = mag[i + w]; }
    else { p = mag[i - w - 1]; q = mag[i + w + 1]; }
    if (m >= p && m >= q) { nms[i] = m; vals.push(m); }
  }
  vals.sort((p, q) => p - q);
  const high = vals.length ? vals[Math.floor(vals.length * 0.9)] : 1, low = high * 0.4;
  const edge = new Uint8Array(n), stack = [];
  for (let i = 0; i < n; i++) if (nms[i] >= high) { edge[i] = 1; stack.push(i); }
  while (stack.length) {
    const i = stack.pop();
    for (const d of [-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1]) {
      const j = i + d;
      if (j >= 0 && j < n && !edge[j] && nms[j] >= low) { edge[j] = 1; stack.push(j); }
    }
  }
  const out = ctx.createImageData(w, h);
  for (let i = 0; i < n; i++) out.data.set(edge[i] ? [255, 255, 255, 255] : [0, 0, 0, 255], 4 * i);
  ctx.putImageData(out, 0, 0);
  return { illum, edges: c.toDataURL('image/png') };
}

function Card({ num, title, src, caption, ratio }) {
  return (
    <figure style={{ margin: 0, background: 'var(--bg-warm-stone-light)', border: '1px solid var(--border-medium)', borderRadius: 18, padding: 'clamp(14px, 2vw, 18px)', display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
      <h2 style={{ ...mono, margin: 0, fontSize: '0.8rem', fontWeight: 400, letterSpacing: '0.08em', color: 'var(--accent-petrol-teal)', textTransform: 'uppercase' }}>{num} / {title}</h2>
      {src
        ? <img src={src} alt={title} style={{ width: '100%', height: 'auto', display: 'block', border: '1px solid var(--border-medium)' }} />
        : <div style={{ width: '100%', aspectRatio: ratio, background: 'var(--bg-warm-stone)', border: '1px solid var(--border-medium)' }} />}
      <figcaption style={{ borderTop: '1px solid var(--border-medium)', paddingTop: 14, fontSize: '0.8rem', color: 'var(--text-ink-black)' }}>{caption}</figcaption>
    </figure>
  );
}

export default function IntermediateRepresentations({ items, onNavigateToDetect }) {
  const [sel, setSel] = useState(0);
  const [maps, setMaps] = useState(null);
  const item = items[Math.min(sel, items.length - 1)];

  useEffect(() => {
    if (!item) return undefined;
    let live = true;
    setMaps(null);
    computeMaps(item.ev.original).then((m) => { if (live) setMaps(m); });
    return () => { live = false; };
  }, [item]);

  const ratio = item ? `${item.ev.display_width} / ${item.ev.display_height}` : '4 / 3';
  return (
    <div className="detect-container">
      <header>
        <div style={{ ...mono, fontSize: '0.9rem', letterSpacing: '0.06em', color: 'var(--accent-petrol-teal)', textTransform: 'uppercase' }}>Visual outputs / 3.2</div>
        <h1 style={{ margin: '6px 0 0', fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: 'clamp(1.8rem, 4vw, 2.6rem)', lineHeight: 1.15 }}>Intermediate representations</h1>
        <div style={{ height: 3, background: 'var(--accent-tangerine)', margin: '14px 0 16px' }} />
        <p style={{ ...mono, margin: 0, fontSize: '0.85rem', color: 'var(--text-ink-muted)' }}>
          {item ? `${item.name} · illumination estimate and edge structure derived from the same image` : 'Input photo · illumination estimate and edge structure derived from the same image'}
        </p>
      </header>

      {!item ? (
        <div style={{ background: 'var(--bg-warm-stone-light)', border: '1px dashed var(--border-dashed)', borderRadius: 18, padding: 'clamp(18px, 4vw, 32px)', display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'flex-start' }}>
          <span style={{ ...mono, fontSize: '0.8rem', color: 'var(--text-ink-muted)' }}>No analysis yet this session. Analyse a photo on the Detect page and its real image will appear here.</span>
          <button className="btn-editorial-teal" onClick={onNavigateToDetect}>
            <span>GO TO DETECTION WORKSPACE</span>
            <ArrowUpRight size={16} strokeWidth={2.4} />
          </button>
        </div>
      ) : (
        <>
          {items.length > 1 && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
              <span style={{ ...mono, fontSize: '0.7rem', color: 'var(--text-ink-muted)' }}>ANALYSIS:</span>
              {items.map((it, i) => (
                <button key={it.id} onClick={() => setSel(i)} aria-pressed={i === sel}
                  style={{ ...mono, fontSize: '0.7rem', cursor: 'pointer', padding: '4px 10px', border: '1px solid var(--text-ink-black)', maxWidth: 190, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', background: i === sel ? 'var(--accent-petrol-teal)' : 'transparent', color: i === sel ? 'var(--bg-warm-chalk)' : 'var(--text-ink-black)' }}>
                  {it.name}
                </button>
              ))}
            </div>
          )}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 'clamp(14px, 2vw, 24px)' }}>
            <Card num="01" title="Input image" src={item.ev.original} ratio={ratio} caption={`Your uploaded photo (${item.ev.original_dimensions?.width}×${item.ev.original_dimensions?.height}, shown at ${item.ev.display_width}×${item.ev.display_height})`} />
            <Card num="02" title="Estimated illumination" src={maps?.illum} ratio={ratio} caption="Smoothed luminance field (Gaussian low-pass)" />
            <Card num="03" title="Edge map" src={maps?.edges} ratio={ratio} caption="Canny gradient boundaries (diagnostic)" />
          </div>
          <div style={{ ...mono, fontSize: '0.75rem', color: 'var(--text-ink-muted)', lineHeight: 1.8 }}>
            <div>NOTE: These are computed diagnostic visualizations from the uploaded image.</div>
            <div>They are not learned reflectance maps or neural attention weights, and are not captured app outputs (the pipeline does not store them).</div>
          </div>
        </>
      )}
    </div>
  );
}

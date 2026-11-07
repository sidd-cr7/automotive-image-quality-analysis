import React, { useEffect, useState } from 'react';
import { ArrowUpRight } from 'lucide-react';

const TAGS = ['Low-light', 'Backlit', 'Other'];
const CREAM = [241, 235, 221];
const INK = [32, 32, 30];
const TEAL = [23, 107, 112];

const mono = { fontFamily: 'var(--font-mono)', fontSize: '0.62rem', letterSpacing: '0.12em', color: 'var(--text-ink-muted)' };
const imgStyle = { width: '100%', height: 'auto', display: 'block', border: '1px solid var(--border-medium)' };

const loadImg = (src) => new Promise((res) => { const i = new Image(); i.onload = () => res(i); i.src = src; });
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

/* Diagnostics computed in the browser from a real pipeline image. These are NOT model outputs. */
async function diagnostics(src) {
  const img = await loadImg(src);
  const w = img.naturalWidth, h = img.naturalHeight;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const px = ctx.getImageData(0, 0, w, h).data;
  const g = new Float32Array(w * h);
  let sum = 0;
  for (let i = 0; i < w * h; i++) { g[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]; sum += g[i]; }

  // Sobel edge magnitude (ink on cream)
  const mag = new Float32Array(w * h);
  let max = 1;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const gx = -g[i - w - 1] - 2 * g[i - 1] - g[i + w - 1] + g[i - w + 1] + 2 * g[i + 1] + g[i + w + 1];
    const gy = -g[i - w - 1] - 2 * g[i - w] - g[i - w + 1] + g[i + w - 1] + 2 * g[i + w] + g[i + w + 1];
    mag[i] = Math.hypot(gx, gy);
    if (mag[i] > max) max = mag[i];
  }
  const edge = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const [r, gg, b] = mix(CREAM, INK, Math.min(1, (mag[i] / max) * 2.2));
    edge.data.set([r, gg, b, 255], 4 * i);
  }
  ctx.putImageData(edge, 0, 0);
  const edgeUrl = c.toDataURL('image/jpeg', 0.85);

  // Illumination estimate: low-pass luminance (heavy blur via a tiny downsample), ink -> teal -> cream
  const tw = 24, th = Math.max(2, Math.round((24 * h) / w));
  const tiny = document.createElement('canvas');
  tiny.width = tw; tiny.height = th;
  const tctx = tiny.getContext('2d', { willReadFrequently: true });
  tctx.drawImage(img, 0, 0, tw, th);
  const td = tctx.getImageData(0, 0, tw, th);
  for (let i = 0; i < tw * th; i++) {
    const l = (0.299 * td.data[4 * i] + 0.587 * td.data[4 * i + 1] + 0.114 * td.data[4 * i + 2]) / 255;
    const col = l < 0.5 ? mix(INK, TEAL, l * 2) : mix(TEAL, CREAM, (l - 0.5) * 2);
    td.data.set([col[0], col[1], col[2], 255], 4 * i);
  }
  tctx.putImageData(td, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(tiny, 0, 0, w, h);
  return { edge: edgeUrl, illum: c.toDataURL('image/jpeg', 0.85), meanLum: sum / (w * h) };
}

function Figure({ src, caption }) {
  return (
    <figure style={{ margin: 0 }}>
      {src ? <img src={src} alt={caption} style={imgStyle} /> : <div style={{ ...imgStyle, aspectRatio: '4 / 3', background: 'var(--bg-warm-stone)' }} />}
      <figcaption style={{ ...mono, marginTop: 6 }}>{caption}</figcaption>
    </figure>
  );
}

function Item({ item }) {
  const [tag, setTag] = useState(null);
  const [d, setD] = useState({});
  useEffect(() => {
    let live = true;
    Promise.all([diagnostics(item.ev.original), diagnostics(item.ev.enhanced)]).then(([o, e]) => {
      if (!live) return;
      setD({ o, e });
      setTag((t) => t || (o.meanLum < 85 ? 'Low-light' : 'Other'));
    });
    return () => { live = false; };
  }, [item]);

  const ev = item.ev;
  const applied = (ev.enhancements_applied || []).map((a) => a.technique).filter(Boolean);
  return (
    <article style={{ background: 'var(--bg-warm-stone-light)', border: '1px solid var(--border-medium)', padding: 'clamp(14px, 3vw, 24px)', display: 'flex', flexDirection: 'column', gap: 18 }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ ...mono, color: 'var(--text-ink-black)', overflowWrap: 'anywhere' }}>{item.name}</span>
        <span style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={mono}>SCENE:</span>
          {TAGS.map((t) => (
            <button key={t} onClick={() => setTag(t)} aria-pressed={tag === t}
              style={{ ...mono, cursor: 'pointer', padding: '4px 8px', border: '1px solid var(--text-ink-black)', background: tag === t ? 'var(--accent-petrol-teal)' : 'transparent', color: tag === t ? 'var(--bg-warm-chalk)' : 'var(--text-ink-black)' }}>
              {t.toUpperCase()}
            </button>
          ))}
        </span>
      </header>

      <div>
        <div style={{ ...mono, color: 'var(--accent-petrol-teal)', marginBottom: 8 }}>PIPELINE OUTPUT · REAL ENHANCER RESULT</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
          <Figure src={ev.original} caption={`BEFORE · ORIGINAL · ${ev.original_dimensions?.width}×${ev.original_dimensions?.height}`} />
          <Figure src={ev.enhanced} caption={`AFTER · ENHANCED · ${ev.enhanced_dimensions?.width}×${ev.enhanced_dimensions?.height}`} />
        </div>
        <div style={{ ...mono, marginTop: 8 }}>
          {ev.enhancement_applied ? `APPLIED: ${applied.join(' + ').toUpperCase()}` : 'NO ENHANCEMENT WAS APPLIED (AFTER = BEFORE)'}
          {typeof item.scoreBefore === 'number' ? ` · QUALITY SCORE ${item.scoreBefore.toFixed(1)} → ${(item.scoreBefore + ev.quality_gain).toFixed(1)}` : ''}
        </div>
      </div>

      <div>
        <div style={{ ...mono, color: 'var(--accent-tangerine)', marginBottom: 4 }}>DIAGNOSTIC VISUALISATIONS · NOT MODEL OUTPUTS</div>
        <p style={{ ...mono, letterSpacing: '0.04em', margin: '0 0 10px', textTransform: 'none' }}>
          Computed in your browser from the two images above (Sobel edge magnitude; low-pass luminance as an illumination estimate). The pipeline does not store these maps.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12 }}>
          <Figure src={d.o?.illum} caption="ILLUMINATION ESTIMATE · BEFORE" />
          <Figure src={d.e?.illum} caption="ILLUMINATION ESTIMATE · AFTER" />
          <Figure src={d.o?.edge} caption="EDGE MAP · BEFORE" />
          <Figure src={d.e?.edge} caption="EDGE MAP · AFTER" />
        </div>
      </div>
      <p style={{ ...mono, letterSpacing: '0.04em', margin: 0, textTransform: 'none' }}>
        Reflectance and attention maps are not implemented in this pipeline, so none are shown.
        {tag === 'Low-light' && d.o ? ` Mean luminance before ${d.o.meanLum.toFixed(0)}/255, after ${d.e?.meanLum.toFixed(0) ?? '…'}/255.` : ''}
        {tag === 'Backlit' ? ' Compare the bright background in both images: it should not clip, while the vehicle body brightens.' : ''}
      </p>
    </article>
  );
}

export default function VisualOutputs({ items, onNavigateToDetect }) {
  return (
    <div className="detect-container">
      <div className="detect-header-block">
        <div className="detect-meta-tag">SECTION 3.2 — VISUAL EVIDENCE</div>
        <h1 className="detect-page-title">Visual Outputs &amp; Intermediate Representations</h1>
        <p className="detect-page-subtext">
          Every photo you analyse this session appears here with the enhancer&apos;s real before and after images. Analyse a
          low-light photo and a backlit photo (for example your Jeep), then mark each one&apos;s scene type.
        </p>
      </div>
      {items.length === 0 ? (
        <div style={{ background: 'var(--bg-warm-stone-light)', border: '1px dashed var(--border-dashed)', padding: 'clamp(18px, 4vw, 32px)', display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'flex-start' }}>
          <span style={mono}>NOTHING ANALYSED YET THIS SESSION</span>
          <button className="btn-editorial-teal" onClick={onNavigateToDetect}>
            <span>GO TO DETECTION WORKSPACE</span>
            <ArrowUpRight size={16} strokeWidth={2.4} />
          </button>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {items.map((it) => <Item key={it.id} item={it} />)}
        </div>
      )}
    </div>
  );
}

import React from 'react';
import { ArrowUpRight } from 'lucide-react';
import '../landing.css';

const ART = '/surreal_hero_artwork.png';
const W = 1024;
const H = 576;
/* Higher-resolution engine region of the approved artwork (source 1200x900). */
const ENG = { src: '/surreal_art_pure.png', sw: 1200, sh: 900, x: 458, y: 286, w: 322, h: 208 };

/* Crops a region (in source pixels) out of the approved hero artwork. */
function Crop({ x, y, w, h, className = '', children, src = ART, sw = W, sh = H, label }) {
  return (
    <div
      className={`ei-crop ${className}`}
      style={{
        aspectRatio: `${w} / ${h}`,
        backgroundImage: `url(${src})`,
        backgroundSize: `${(sw / w) * 100}% ${(sh / h) * 100}%`,
        backgroundPosition: `${(x / (sw - w)) * 100}% ${(y / (sh - h)) * 100}%`,
      }}
      role={label ? 'img' : undefined}
      aria-label={label}
    >
      {children}
    </div>
  );
}

function SectionHead({ num, kicker, title, children }) {
  return (
    <header className="ei-head">
      <span className="ei-bignum" aria-hidden="true">{num}</span>
      <div className="ei-head-text">
        <span className="ei-mono ei-kicker">{kicker}</span>
        <h2 className="ei-h2">{title}</h2>
        <p className="ei-lead">{children}</p>
      </div>
    </header>
  );
}

export default function LandingView({ onNavigateToDetect }) {
  return (
    <div className="landing-container ei-landing">
      {/* ---------- NAV ---------- */}
      <header className="landing-top-nav">
        <span className="brand-spec-tag">FIG. 01 / SYSTEM OVERVIEW</span>
        <span className="meta-category">
          AUTOMOTIVE COMPUTER VISION <span className="meta-dash">—</span> FORENSIC METROLOGY
        </span>
        <button onClick={onNavigateToDetect} className="btn-editorial-teal btn-compact" title="Open Detection Workspace">
          <span>DETECT WORKSPACE</span>
          <ArrowUpRight size={15} strokeWidth={2.4} />
        </button>
      </header>

      {/* ---------- HERO ---------- */}
      <section className="landing-hero-section">
        <div className="hero-header-block">
          <div className="hero-title-wrapper">
            <h1 className="hero-title-line">ENGINE</h1>
            <h1 className="hero-title-line">INTELLIGENCE</h1>
          </div>
          <div className="hero-subtitle-triad">
            <span className="triad-item">DETECT</span><span className="triad-sep">/</span>
            <span className="triad-item">IDENTIFY</span><span className="triad-sep">/</span>
            <span className="triad-item">ANALYZE</span>
          </div>
        </div>
        <div className="hero-artwork-frame">
          <img
            src="/surreal_hero_artwork.jpg"
            alt="Classical sculpture wearing an engine visor beneath a red sun, mountains and a winding road"
            className="hero-artwork-img"
            loading="eager"
          />
        </div>
        <div className="hero-editorial-footer">
          <div className="hero-manifesto-text">
            <p className="hero-lead">
              Vehicle classification, 23-class exterior segmentation and engine-bay reasoning,
              run as three synchronized vision branches on every photograph you upload.
            </p>
          </div>
          <div className="hero-cta-action">
            <button onClick={onNavigateToDetect} className="btn-editorial-teal btn-large">
              <span>ENTER DETECTION WORKSPACE</span>
              <ArrowUpRight size={18} strokeWidth={2.4} />
            </button>
          </div>
        </div>
      </section>

      {/* ---------- 02 THREE BRANCHES ---------- */}
      <section className="ei-section">
        <SectionHead num="02" kicker="FIG. 02 — THREE-BRANCH PERCEPTION ENGINE" title={<>One photograph,<br /><em>three readings.</em></>}>
          A single model cannot find the car, trace its panels and diagnose the engine bay.
          Three specialised branches run in parallel and merge into one report.
        </SectionHead>

        {/* Branch 01 — vehicle id: big photo left, bbox annotations */}
        <article className="ei-branch ei-b1">
          <div className="ei-b1-visual">
            <Crop x={285} y={425} w={360} h={151} className="ei-b1-photo">
              <span className="ei-bbox" style={{ left: '31%', top: '15%', width: '35%', height: '36%' }}>
                <i className="ei-tag ei-tag-teal">VEHICLE</i>
                <b className="ei-corner tl" /><b className="ei-corner tr" /><b className="ei-corner bl" /><b className="ei-corner br" />
              </span>
              <span className="ei-cross" style={{ left: '48.5%', top: '33%' }} />
            </Crop>
            <div className="ei-ruler"><span>x₀</span><i /><span className="ei-mono">LOCALISE → CROP → CLASSIFY</span><i /><span>x₁</span></div>
            <p className="ei-caption ei-mono">FIG. 02.1 — BOUNDING-BOX LOCALISATION</p>
          </div>
          <div className="ei-b1-copy">
            <span className="ei-branchnum" aria-hidden="true">01</span>
            <span className="ei-mono ei-pill ei-pill-teal">BRANCH 01 · VEHICLE IDENTIFICATION</span>
            <h3 className="ei-h3">Find the car. <em>Name it.</em></h3>
            <p className="ei-prose">
              A YOLOv8 detector isolates the vehicle from the background. The crop goes to a ResNet-50
              classifier trained on the Stanford Cars taxonomy, which returns a ranked top-5 list
              rather than a single guess.
            </p>
            <dl className="ei-spec">
              <div><dt>Detector</dt><dd>YOLOv8</dd></div>
              <div><dt>Classifier</dt><dd>ResNet-50</dd></div>
              <div><dt>Classes</dt><dd>196</dd></div>
              <div><dt>Output</dt><dd>Top-5 ranking</dd></div>
            </dl>
          </div>
        </article>

        {/* Branch 02 — exterior: centred schematic with leader labels */}
        <article className="ei-branch ei-b2">
          <div className="ei-b2-top">
            <span className="ei-branchnum ei-branchnum-teal" aria-hidden="true">02</span>
            <div>
              <span className="ei-mono ei-pill ei-pill-tangerine">BRANCH 02 · EXTERIOR INTELLIGENCE</span>
              <h3 className="ei-h3">Every panel, <em>traced.</em></h3>
            </div>
            <p className="ei-prose">
              A YOLO11n-seg network outlines 23 exterior components pixel by pixel: bumpers, lamps,
              glass, doors, wheels. Each mask can be inspected on its own.
            </p>
          </div>
          <div className="ei-b2-stage">
            <ul className="ei-legend ei-legend-l">
              <li><b>01</b> Hood &amp; bonnet</li>
              <li><b>02</b> Front fascia</li>
              <li><b>03</b> Headlamps</li>
            </ul>
            <div className="ei-b2-figure">
              <svg viewBox="0 0 580 230" role="img" aria-label="Segmentation schematic of a coupe body">
                <defs>
                  <pattern id="ei-grid" width="20" height="20" patternUnits="userSpaceOnUse">
                    <path d="M20 0H0V20" fill="none" stroke="rgba(32,32,30,.08)" strokeWidth=".8" />
                  </pattern>
                </defs>
                <rect width="580" height="230" fill="url(#ei-grid)" />
                <path d="M60 170C65 155 80 145 110 140C140 135 190 132 230 95C270 58 330 55 420 65C470 72 500 100 530 130C550 145 560 158 555 175L530 180C515 150 470 150 455 180L190 180C175 150 130 150 115 180Z" fill="rgba(32,32,30,.05)" stroke="#20201E" strokeWidth="2" />
                <path d="M62 168C65 155 80 145 110 140C140 135 190 132 225 105L205 165Z" fill="rgba(23,107,112,.32)" stroke="#176B70" strokeWidth="1.6" />
                <path d="M230 98L415 68C455 76 480 100 495 130L220 133Z" fill="rgba(242,139,69,.28)" stroke="#F28B45" strokeWidth="1.6" />
                <path d="M425 70C475 78 505 105 530 132C550 145 558 158 554 175L525 178C510 152 475 152 460 178L415 178Z" fill="rgba(23,107,112,.2)" stroke="#176B70" strokeWidth="1.4" />
                <circle cx="152" cy="178" r="32" fill="#DED5C5" stroke="#20201E" strokeWidth="2" />
                <circle cx="152" cy="178" r="17" fill="#151513" stroke="#F28B45" strokeWidth="1.5" />
                <circle cx="488" cy="178" r="32" fill="#DED5C5" stroke="#20201E" strokeWidth="2" />
                <circle cx="488" cy="178" r="17" fill="#151513" stroke="#F28B45" strokeWidth="1.5" />
                <g fontFamily="Space Mono, monospace" fontSize="9" fontWeight="700">
                  <polyline points="130,145 110,40 70,40" stroke="#176B70" fill="none" /><circle cx="130" cy="145" r="3" fill="#176B70" /><text x="12" y="36" fill="#176B70">MASK 01 HOOD</text>
                  <polyline points="330,90 330,25 390,25" stroke="#F28B45" fill="none" /><circle cx="330" cy="90" r="3" fill="#F28B45" /><text x="394" y="29" fill="#20201E">MASK 05 GLASS</text>
                  <polyline points="488,178 488,222 430,222" stroke="#20201E" fill="none" /><text x="320" y="225" fill="#20201E">MASK 04 WHEEL</text>
                </g>
              </svg>
            </div>
            <ul className="ei-legend ei-legend-r">
              <li><b>04</b> Wheels &amp; arches</li>
              <li><b>05</b> Greenhouse</li>
              <li><b>…</b> 18 more classes</li>
            </ul>
          </div>
          <p className="ei-caption ei-mono">FIG. 02.2 — ILLUSTRATIVE SCHEMATIC · NOT A MODEL OUTPUT</p>
        </article>

        {/* Branch 03 — engine: full-bleed charcoal band, huge engine image */}
        <article className="ei-branch ei-b3">
          <div className="ei-b3-copy">
            <span className="ei-branchnum ei-branchnum-dark" aria-hidden="true">03</span>
            <span className="ei-mono ei-pill ei-pill-light">BRANCH 03 · INTERIOR &amp; ENGINE REASONING</span>
            <h3 className="ei-h3 ei-h3-light">Read the <em>machine.</em></h3>
            <p className="ei-prose ei-prose-light">
              Engine bays and cockpits do not fit a fixed class list. Branch 03 sends a downscaled
              frame to a Qwen vision model on Groq, with a second instance on hot standby, and returns
              a written mechanical diagnosis.
            </p>
            <dl className="ei-spec ei-spec-light">
              <div><dt>Model</dt><dd>Qwen vision (Groq)</dd></div>
              <div><dt>Failover</dt><dd>Secondary instance</dd></div>
              <div><dt>Input</dt><dd>512 px frame</dd></div>
            </dl>
          </div>
          <div className="ei-b3-visual">
            <Crop {...ENG} className="ei-b3-engine" label="Red engine block collage" />
            <span className="ei-callout" style={{ left: '8%', top: '12%' }}><i /> VALVE COVER</span>
            <span className="ei-callout" style={{ right: '4%', top: '62%' }}><i /> EXHAUST RUNNERS</span>
            <span className="ei-callout" style={{ left: '14%', bottom: '8%' }}><i /> ACCESSORY DRIVE</span>
          </div>
        </article>
      </section>

      {/* ---------- 03 QUALITY ---------- */}
      <section className="ei-section">
        <SectionHead num="03" kicker="FIG. 03 — QUALITY ASSESSMENT" title={<>Fix the photo<br /><em>before</em> reading it.</>}>
          Field photos are blurred, over-exposed or buried in shadow. Every frame is measured first,
          then repaired only if it needs it.
        </SectionHead>

        <div className="ei-q">
          <figure className="ei-q-a">
            <div className="ei-split">
              <Crop {...ENG} className="ei-before ei-blurred" />
              <Crop {...ENG} className="ei-after" />
              <span className="ei-split-line" />
              <span className="ei-mono ei-split-l">BEFORE · BLURRED</span>
              <span className="ei-mono ei-split-r">AFTER · SHARPENED</span>
            </div>
            <figcaption className="ei-mono">FIG. 03.1 — ILLUSTRATIVE COMPARISON · BLUR SIMULATED WITH A CSS FILTER</figcaption>
          </figure>

          <figure className="ei-q-b">
            <div className="ei-split">
              <Crop x={285} y={425} w={360} h={151} className="ei-before ei-dark" />
              <Crop x={285} y={425} w={360} h={151} className="ei-after ei-clahe" />
              <span className="ei-split-line" />
              <span className="ei-mono ei-split-l">BEFORE · UNDEREXPOSED</span>
              <span className="ei-mono ei-split-r">AFTER · CLAHE</span>
            </div>
            <figcaption className="ei-mono">FIG. 03.2 — ILLUSTRATIVE COMPARISON · DARKENING SIMULATED WITH A CSS FILTER</figcaption>
          </figure>

          <div className="ei-q-c">
            <div className="ei-hist">
            <svg viewBox="0 0 220 90" aria-label="Illustrative luminance histograms">
              <g fill="#20201E" opacity=".75">
                {[4, 9, 18, 34, 52, 60, 44, 26, 12, 6, 3, 2].map((v, i) => <rect key={i} x={4 + i * 8} y={80 - v} width="6" height={v} />)}
              </g>
              <g fill="#176B70">
                {[12, 22, 30, 36, 40, 42, 40, 36, 30, 22, 14, 8].map((v, i) => <rect key={i} x={116 + i * 8} y={80 - v} width="6" height={v} />)}
              </g>
              <line x1="0" y1="81" x2="220" y2="81" stroke="#20201E" strokeWidth=".8" />
              <text x="4" y="10" fontFamily="Space Mono, monospace" fontSize="7" fill="#20201E">RAW</text>
              <text x="116" y="10" fontFamily="Space Mono, monospace" fontSize="7" fill="#176B70">EQUALISED</text>
            </svg>
            <p className="ei-mono">ILLUSTRATIVE ONLY · NOT MEASURED DATA</p>
            </div>
            <ol className="ei-steps">
              <li><b>01</b> Measure blur (Laplacian variance); below 100 triggers sharpening</li>
              <li><b>02</b> Measure exposure and dynamic range</li>
              <li><b>03</b> Enhance, then keep whichever frame carries more Shannon entropy</li>
            </ol>
          </div>
        </div>
      </section>

      {/* ---------- 04 SPECS & VALUATION ---------- */}
      <section className="ei-section">
        <SectionHead num="04" kicker="FIG. 04 — SPECIFICATIONS & MARKET VALUATION" title={<>From a name<br />to a <em>dossier.</em></>}>
          A confirmed make and model unlocks factory specifications and an indicative resale range.
          When the model or year is unknown, those fields stay empty. Nothing is invented.
        </SectionHead>

        <div className="ei-spec-layout">
          <div className="ei-spec-engine">
            <div className="ei-spec-sun" />
            <Crop {...ENG} label="Engine illustration" />
            <span className="ei-measure ei-measure-h"><i /><b className="ei-mono">LENGTH</b><i /></span>
            <span className="ei-measure ei-measure-v"><i /><b className="ei-mono">HEIGHT</b><i /></span>
            <p className="ei-caption ei-mono">FIG. 04.1 — POWERTRAIN · ARTWORK</p>
          </div>

          <div className="ei-ledger">
            <h3 className="ei-h3">Powertrain &amp; architecture</h3>
            {[
              ['Displacement', 'Litres, cylinder count'],
              ['Induction', 'Aspirated or forced'],
              ['Valve train', 'Cams, valves per cylinder'],
              ['Transmission', 'Type and gear count'],
              ['Drive', 'FWD · RWD · AWD'],
              ['Body', 'Coupe · sedan · hatch · wagon'],
            ].map(([k, v], i) => (
              <div className="ei-ledger-row" key={k}>
                <span className="ei-mono ei-ledger-i">{String(i + 1).padStart(2, '0')}</span>
                <span className="ei-ledger-k">{k}</span>
                <span className="ei-ledger-v">{v}</span>
              </div>
            ))}
          </div>

          <div className="ei-val">
            <span className="ei-mono ei-kicker">SECONDARY-MARKET RANGE</span>
            <div className="ei-val-scale">
              <span><b className="ei-mono">LOW</b>Fair condition</span>
              <span className="ei-val-mid"><b className="ei-mono">MEDIAN</b>Typical wear</span>
              <span><b className="ei-mono">HIGH</b>Pristine</span>
            </div>
            <p className="ei-prose">
              Depreciation is modelled from vehicle age and a 12,000 km/year baseline, using listings
              from CarWale, CarDekho, Parkers and Kelley Blue Book. Currencies: ₹, $, £.
            </p>
          </div>
        </div>
      </section>

      {/* ---------- 05 CTA ---------- */}
      <section className="ei-cta">
        <div className="ei-cta-art" aria-hidden="true">
          <span className="ei-cta-disc" />
          <Crop x={400} y={170} w={250} h={330} className="ei-cta-face" />
          <Crop x={0} y={460} w={290} h={116} className="ei-cta-mtn" />
        </div>
        <div className="ei-cta-copy">
          <span className="ei-mono ei-cta-kicker">05 — BEGIN</span>
          <h2 className="ei-cta-title">Upload a car.<br /><em>Read the machine.</em></h2>
          <p className="ei-cta-text">Exterior, engine bay or cockpit. One photograph is enough.</p>
          <button onClick={onNavigateToDetect} className="ei-cta-btn">
            <span>LAUNCH DETECTION WORKSPACE</span>
            <ArrowUpRight size={20} strokeWidth={2.4} />
          </button>
        </div>
      </section>
    </div>
  );
}

import React, { useEffect, useState } from 'react';
import {
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  Layers,
  Fuel,
  DollarSign,
  Activity,
  AlertCircle,
  Eye,
  Cpu,
  Zap,
  Sparkles
} from 'lucide-react';

export default function ResultsArea({ resultData, isLoading, analysis }) {
  const [showExtended, setShowExtended] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const startedAt = analysis?.startedAt;
  useEffect(() => {
    if (!isLoading || !startedAt) return undefined;
    const tick = () => setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    tick();
    const id = setInterval(tick, 500);
    return () => clearInterval(id);
  }, [isLoading, startedAt]);

  // ============================================================
  // 1. LOADING STATE
  // ============================================================
  if (isLoading) {
    const stages = [
      { key: 'upload', label: 'Uploading', done: !!analysis?.uploadDone, active: !analysis?.uploadDone, note: analysis?.uploadDone ? '' : `${analysis?.uploadPct ?? 0}%` },
      { key: 'detection', label: 'Vehicle Detection', hint: 'includes image-quality checks' },
      { key: 'inspection', label: 'Visual Inspection', hint: 'Gemini, then Groq, then local models' },
      { key: 'final', label: 'Final Analysis', hint: 'verdict, specs, valuation' },
    ].map((st) => {
      if (st.key === 'upload') return st;
      const done = analysis?.completed?.includes(st.key);
      return { ...st, done: !!done, active: !done && analysis?.current === st.key };
    });
    return (
      <div style={styles.container}>
        <div style={styles.card}>
          <div style={styles.headerRow}>
            <span style={styles.subHeader}>ANALYSIS IN PROGRESS</span>
            <span style={styles.loadingPulseTag}>
              <span style={styles.pulseDot} />
              {elapsed}s
            </span>
          </div>
          <div style={styles.loadingBody}>
            <h3 style={styles.loadingTitle}>Analysing your image</h3>
            <div style={styles.branchStatusList}>
              {stages.map((st) => (
                <div key={st.key} style={{ ...styles.branchStatusItem, opacity: st.done || st.active ? 1 : 0.55 }}>
                  <span style={{ width: 10, height: 10, borderRadius: '50%', flexShrink: 0, background: st.done ? 'var(--accent-petrol-teal)' : st.active ? 'var(--accent-tangerine)' : 'transparent', border: '1.5px solid var(--text-ink-black)' }} />
                  <span style={styles.branchStatusLabel}>
                    {st.label}
                    {st.hint && <span style={{ display: 'block', fontSize: '0.68rem', opacity: 0.65 }}>{st.hint}</span>}
                  </span>
                  <span style={styles.branchStatusState}>{st.done ? 'Done' : st.active ? (st.note || 'Running') : 'Waiting'}</span>
                </div>
              ))}
            </div>
            <p style={{ ...styles.loadingDesc, marginTop: 14 }}>
              Stages update from the server as each one finishes.
              {elapsed >= 45 && ' This is taking longer than usual, possibly because the vision service is rate-limited. The request is still running.'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ============================================================
  // 2. EMPTY / INTRODUCTORY STATE (NO FABRICATED DATA)
  // ============================================================
  if (!resultData) {
    return (
      <div style={styles.container}>
        <div style={styles.emptyCard}>
          <div style={styles.headerRow}>
            <span style={styles.subHeader}>DETECTION RESULTS</span>
            <span style={styles.standbyBadge}>STANDBY</span>
          </div>

          <div style={styles.emptyBody}>
            <div style={styles.emptyIconCircle}>
              <Activity size={28} color="var(--accent-petrol-teal)" />
            </div>

            <h3 style={styles.emptyTitle}>Awaiting Imagery</h3>
            <p style={styles.emptyDesc}>
              Upload a vehicle exterior, engine bay, or interior image in the workspace to
              initiate the multi-model intelligence pipeline.
            </p>

            <div style={styles.standbyBranchList}>
              <div style={styles.standbyBranchItem}>
                <div style={styles.standbyBranchDot} />
                <span>Branch 1 — Vehicle Make / Model / Year</span>
              </div>
              <div style={styles.standbyBranchItem}>
                <div style={styles.standbyBranchDot} />
                <span>Branch 2 — 23 Exterior Components (YOLO11n-seg)</span>
              </div>
              <div style={styles.standbyBranchItem}>
                <div style={styles.standbyBranchDot} />
                <span>Branch 3 — Multimodal Engine & Cockpit Reasoning</span>
              </div>
            </div>

            <div style={styles.emptyHint}>
              <span>Awaiting input from the upload module on the left.</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ============================================================
  // 3. ACTUAL LIVE BACKEND PREDICTIONS
  // ============================================================
  const vehicle = resultData.vehicle || {};
  const specs = resultData.vehicle_specifications || {};
  const engine = specs.engine || {};
  const valuation = resultData.market_valuation || {};
  const exterior = resultData.exterior_analysis || {};
  const quality = resultData.image_quality || {};
  const originalB1 = resultData.original_model_outputs?.branch_1_vehicle_id || {};
  const topCandidates = originalB1?.predictions?.[0]?.top5 || [];
  const isUncertain = vehicle.is_uncertain === true;
  const idSource = vehicle.identification_source;
  const isGroqId = typeof idSource === 'string' && idSource.endsWith('_vision');
  const providerName = vehicle.vision_provider ? vehicle.vision_provider.charAt(0).toUpperCase() + vehicle.vision_provider.slice(1) : 'vision';

  const displayMake = vehicle.make || specs.make || 'Unidentified Vehicle';
  const modelKnown = vehicle.model_known !== false && !!vehicle.model;
  const nonExterior = resultData.scene_type === 'engine_bay' || resultData.scene_type === 'interior';
  const displayModel = modelKnown ? vehicle.model : (nonExterior ? 'Not identifiable from this view' : 'Not identified (manufacturer only)');
  const displayConfidence = vehicle.confidence !== undefined ? vehicle.confidence : null;
  const confidencePercent = displayConfidence !== null ? (displayConfidence * 100).toFixed(1) : null;

  const displayType = engine.configuration || specs.body_type || 'Unavailable';
  const displayDisplacement = engine.displacement || 'Unavailable';

  // Format market valuation price
  let formattedPrice = null;
  if (valuation?.estimated_price) {
    if (valuation.currency === 'INR') {
      const lakhs = (valuation.estimated_price / 100000).toFixed(1);
      formattedPrice = `₹ ${lakhs} Lakhs (${valuation.market_region || 'India'})`;
    } else {
      formattedPrice = `${valuation.currency || '$'} ${valuation.estimated_price.toLocaleString()}`;
    }
  }

  // Brand emblem generator/lookup
  const renderBrandBadge = (brandName) => {
    const brand = (brandName || '').toLowerCase();
    if (brand.includes('toyota')) {
      return (
        <svg width="32" height="24" viewBox="0 0 32 24" fill="none" stroke="currentColor" strokeWidth="1.6">
          <ellipse cx="16" cy="12" rx="14" ry="10" />
          <ellipse cx="16" cy="11" rx="8" ry="5.5" />
          <ellipse cx="16" cy="13.5" rx="3.5" ry="7.5" />
        </svg>
      );
    } else if (brand.includes('bmw')) {
      return (
        <svg width="30" height="30" viewBox="0 0 30 30" fill="none" stroke="currentColor" strokeWidth="1.6">
          <circle cx="15" cy="15" r="13" />
          <path d="M15 2V28M2 15H28" />
        </svg>
      );
    } else if (brand.includes('audi')) {
      return (
        <svg width="38" height="20" viewBox="0 0 38 20" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="8" cy="10" r="6" />
          <circle cx="14" cy="10" r="6" />
          <circle cx="20" cy="10" r="6" />
          <circle cx="26" cy="10" r="6" />
        </svg>
      );
    } else if (brand.includes('honda')) {
      return (
        <svg width="28" height="24" viewBox="0 0 28 24" fill="none" stroke="currentColor" strokeWidth="1.8">
          <rect x="2" y="2" width="24" height="20" rx="3" />
          <path d="M7 6V18M21 6V18M7 12H21" />
        </svg>
      );
    }
    return (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="14" cy="14" r="12" />
        <path d="M14 6L14 22M6 14L22 14" />
      </svg>
    );
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        {/* Card Header Label */}
        <div style={styles.headerRow}>
          <span style={styles.subHeader}>DETECTION RESULT</span>
          {specs.verification_status && (
            <span style={styles.verifiedStatusBadge}>
              <ShieldCheck size={12} color="var(--accent-petrol-teal)" />
              <span>{specs.verification_status.replace(/_/g, ' ').toUpperCase()}</span>
            </span>
          )}
        </div>

        {resultData.enhancement_visuals?.original && resultData.enhancement_visuals?.enhanced && (() => {
          const ev = resultData.enhancement_visuals;
          const techniques = (ev.enhancements_applied || []).map((e) => e.technique || e.name).filter(Boolean);
          const label = { fontFamily: 'var(--font-mono)', fontSize: '0.62rem', letterSpacing: '0.12em', color: 'var(--text-ink-muted)' };
          const img = { width: '100%', height: 'auto', display: 'block', aspectRatio: `${ev.display_width} / ${ev.display_height}`, objectFit: 'cover', border: '1px solid var(--border-medium)' };
          return (
            <div style={{ margin: '4px 0 18px' }}>
              <div style={{ ...label, marginBottom: 8 }}>IMAGE ENHANCEMENT · BEFORE / AFTER</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                <figure style={{ margin: 0 }}>
                  <img src={ev.original} alt="Original uploaded image" style={img} />
                  <figcaption style={label}>BEFORE · ORIGINAL · {ev.original_dimensions?.width}×{ev.original_dimensions?.height}</figcaption>
                </figure>
                <figure style={{ margin: 0 }}>
                  <img src={ev.enhanced} alt="Enhanced image from the pipeline" style={img} />
                  <figcaption style={label}>AFTER · ENHANCED · {ev.enhanced_dimensions?.width}×{ev.enhanced_dimensions?.height}</figcaption>
                </figure>
              </div>
              <div style={{ ...label, marginTop: 8 }}>
                {ev.enhancement_applied ? `YOUR PHOTO NEEDED FIXING. APPLIED: ${techniques.join(' + ').toUpperCase() || 'ADAPTIVE ENHANCEMENT'}${typeof resultData.image_quality?.overall_score === 'number' ? ` · QUALITY SCORE ${resultData.image_quality.overall_score.toFixed(1)} → ${(resultData.image_quality.overall_score + ev.quality_gain).toFixed(1)}` : ''}` : 'PHOTO QUALITY WAS FINE: NO ENHANCEMENT WAS NEEDED (AFTER = BEFORE)'}
              </div>
            </div>
          );
        })()}

        {/* Brand Name & Logo Badge */}
        {isUncertain && (
          <span style={{ display: 'inline-block', fontFamily: 'var(--font-mono)', fontSize: '0.66rem', letterSpacing: '0.14em', fontWeight: 700, background: 'var(--accent-tangerine)', color: 'var(--text-ink-black)', padding: '3px 8px', marginBottom: 6 }}>
            {isGroqId ? `AI-IDENTIFIED (${providerName.toUpperCase()}) · UNVERIFIED` : 'UNCONFIRMED · POSSIBLE MATCH'}
          </span>
        )}
        <div style={styles.brandRow}>
          <h2 style={styles.makeTitle}>{isUncertain && displayMake !== 'Unidentified vehicle' ? `${displayMake}?` : displayMake}</h2>
          {!isUncertain && (
            <div style={styles.brandLogoCircle}>
              {renderBrandBadge(displayMake)}
            </div>
          )}
        </div>

        {/* Confidence Section (Only rendered if model returned a score) */}
        {confidencePercent !== null && (
          <div style={styles.confidenceSection}>
            <div style={styles.confidenceTag}>
              <span>Confidence</span>
            </div>
            <div style={styles.percentageRow}>
              <span style={styles.percentageText}>{confidencePercent}%</span>
            </div>

            {/* Confidence Progress Bar in Petrol Teal */}
            <div style={styles.barTrack}>
              <div
                style={{
                  ...styles.barFill,
                  width: `${Math.min(100, Math.max(0, displayConfidence * 100))}%`,
                }}
              />
            </div>
          </div>
        )}

        {isUncertain && (
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', letterSpacing: '0.1em', color: 'var(--accent-petrol-teal)', borderLeft: '3px solid var(--accent-tangerine)', padding: '6px 10px', margin: '0 0 14px' }}>
            {isGroqId ? `IDENTIFIED BY THE ${providerName.toUpperCase()} VISION MODEL, NOT THE CLASSIFIER. THERE IS NO CALIBRATED CONFIDENCE SCORE. TREAT IT, AND ANY SPECS OR PRICE BELOW, AS UNVERIFIED.` : 'LOW CONFIDENCE. THE MODEL IS NOT SURE. TREAT THIS IDENTIFICATION, AND ANY SPECS OR PRICE BELOW, AS UNVERIFIED.'}
          </p>
        )}

        {(isGroqId || vehicle.identity_conflict) && vehicle.classifier_top1 && (
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.64rem', letterSpacing: '0.08em', color: 'var(--text-ink-muted)', margin: '0 0 14px' }}>
            {vehicle.identity_conflict ? 'MODELS DISAGREE. CLASSIFIER SAYS' : "CLASSIFIER'S OWN GUESS"}: {vehicle.classifier_top1.toUpperCase()} ({(vehicle.classifier_confidence * 100).toFixed(1)}%)
          </p>
        )}
        {vehicle.identification_note && !isUncertain && (
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.64rem', letterSpacing: '0.08em', color: 'var(--text-ink-muted)', margin: '0 0 14px' }}>
            {vehicle.identification_note.toUpperCase()}
          </p>
        )}

        {(() => {
          const scene = resultData.scene_type;
          const eng = resultData.engine_details || {};
          const engInfo = resultData.visible_engine_info || {};
          const inter = resultData.interior_analysis || {};
          const showEngine = scene === 'engine_bay' || scene === 'mixed' || engInfo.is_engine_visible;
          const showInterior = scene === 'interior' || scene === 'mixed' || inter.is_interior_visible;
          if (!showEngine && !showInterior) return null;
          const lbl = { fontFamily: 'var(--font-mono)', fontSize: '0.64rem', letterSpacing: '0.1em', color: 'var(--text-ink-muted)' };
          const row = { display: 'flex', gap: 8, justifyContent: 'space-between', borderTop: '1px solid var(--border-subtle)', padding: '6px 0', fontSize: '0.8rem' };
          const seen = (list) => (list || []).filter((c) => c && c.visible !== false && c.component);
          const engParts = seen(engInfo.engine_components);
          const intParts = seen(inter.interior_components);
          return (
            <div style={{ border: '1px solid var(--border-medium)', background: 'rgba(241,235,221,0.5)', padding: '12px 14px', margin: '0 0 16px' }}>
              <div style={{ ...lbl, color: 'var(--accent-petrol-teal)', fontWeight: 700, marginBottom: 8 }}>
                {scene === 'engine_bay' ? 'ENGINE BAY VIEW' : scene === 'interior' ? 'INTERIOR VIEW' : 'ENGINE / INTERIOR FINDINGS'} · AI-ESTIMATED FROM THE PHOTO · UNVERIFIED
              </div>
              {showEngine && (
                <div style={{ marginBottom: showInterior ? 12 : 0 }}>
                  {(eng.configuration || eng.cylinders) && (
                    <div style={row}><span style={lbl}>ENGINE</span><strong>{[eng.configuration, eng.cylinders ? `${eng.cylinders} cylinders` : null].filter(Boolean).join(' · ')}</strong></div>
                  )}
                  {eng.cylinder_count_evidence?.length > 0 && <div style={row}><span style={lbl}>HOW COUNTED</span><span style={{ textAlign: 'right' }}>{eng.cylinder_count_evidence.join('; ')}</span></div>}
                  {eng.aspiration && <div style={row}><span style={lbl}>ASPIRATION</span><span>{eng.aspiration}</span></div>}
                  {eng.manufacturer_cues?.length > 0 && <div style={row}><span style={lbl}>LOGOS / TEXT SEEN</span><span style={{ textAlign: 'right' }}>{eng.manufacturer_cues.join(', ')}</span></div>}
                  {engParts.length > 0 && <div style={row}><span style={lbl}>VISIBLE PARTS</span><span style={{ textAlign: 'right' }}>{engParts.map((c) => c.component).join(', ')}</span></div>}
                  {eng.notes && <div style={{ ...row, flexDirection: 'column' }}><span style={lbl}>NOTES</span><span>{eng.notes}</span></div>}
                </div>
              )}
              {showInterior && (
                <div>
                  {intParts.length > 0 && <div style={row}><span style={lbl}>VISIBLE</span><span style={{ textAlign: 'right' }}>{intParts.map((c) => c.component).join(', ')}</span></div>}
                  {intParts.filter((c) => c.details && !/not visible/i.test(c.details)).slice(0, 5).map((c) => (
                    <div key={c.component} style={row}><span style={lbl}>{String(c.component).toUpperCase()}</span><span style={{ textAlign: 'right' }}>{c.details}</span></div>
                  ))}
                  {inter.identity_cues?.length > 0 && <div style={row}><span style={lbl}>BRAND / MODEL CUES</span><span style={{ textAlign: 'right' }}>{inter.identity_cues.join('; ')}</span></div>}
                </div>
              )}
              {(engInfo.notes || inter.notes) && <div style={{ ...lbl, marginTop: 8, letterSpacing: '0.04em', textTransform: 'none' }}>{engInfo.notes || inter.notes}</div>}
            </div>
          );
        })()}

        {/* Spec Metrics Row */}
        <div style={styles.metricsGrid}>
          <div style={styles.metricItem}>
            <span style={styles.metricLabel}>{isUncertain || !modelKnown ? 'Model (unconfirmed)' : 'Model / Trim'}</span>
            <span style={styles.metricValue}>{displayModel}</span>
          </div>

          <div style={styles.metricDivider} />

          <div style={styles.metricItem}>
            <span style={styles.metricLabel}>Engine / Type</span>
            <span style={styles.metricValue}>{displayType}</span>
          </div>

          <div style={styles.metricDivider} />

          <div style={styles.metricItem}>
            <span style={styles.metricLabel}>Displacement</span>
            <span style={styles.metricValue}>{displayDisplacement}</span>
          </div>
        </div>

        {specs.variants?.length > 0 && (
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.64rem', letterSpacing: '0.08em', color: 'var(--text-ink-muted)', margin: '0 0 14px' }}>
            DATABASE (US EPA) · {specs.variants.length} CONFIGURATION{specs.variants.length > 1 ? 'S' : ''} FOR THIS MODEL YEAR; THE EXACT ONE IS NOT IDENTIFIABLE FROM A PHOTO: {specs.variants.map((v) => `${v.displacement_l ?? '?'}L ${v.cylinders ?? '?'}-CYL ${v.transmission ?? ''}`.toUpperCase()).join(' · ')}
          </p>
        )}

        {/* Real Top Predictions Ranking (if present in output) */}
        {topCandidates && topCandidates.length > 0 && (
          <div style={styles.topPredictionsContainer}>
            <span style={styles.topPredHeader}>TOP PREDICTIONS</span>
            <div style={styles.predList}>
              {topCandidates.slice(0, 4).map((cand, idx) => {
                const confVal = cand.confidence !== undefined ? cand.confidence : 0;
                const pct = (confVal * 100).toFixed(1);
                return (
                  <div key={idx} style={styles.predRow}>
                    <span style={styles.predName}>{cand.label || cand.class_name || `Class ${idx}`}</span>
                    <div style={styles.predBarTrack}>
                      <div
                        style={{
                          ...styles.predBarFill,
                          width: `${Math.min(100, confVal * 100)}%`,
                          backgroundColor: idx === 0 ? 'var(--accent-petrol-teal)' : 'var(--bg-warm-stone-dark)',
                        }}
                      />
                    </div>
                    <span style={styles.predPct}>{pct}%</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Market Valuation (if present in output) */}
        {formattedPrice && (
          <div style={styles.valuationBadge}>
            <DollarSign size={15} color="var(--accent-petrol-teal)" />
            <span style={styles.valuationText}>Estimated Market Value: <strong>{formattedPrice}</strong>{isUncertain ? ' (unverified: based on an uncertain identification)' : ''}</span>
          </div>
        )}

        {!formattedPrice && (
          <div style={styles.valuationBadge}>
            <DollarSign size={15} color="var(--accent-tangerine)" />
            <span style={styles.valuationText}>
              <strong>Market valuation unavailable — insufficient verified market data.</strong> {valuation.assumptions?.[0]?.replace(/^Market (value|valuation) unavailable:\s*(insufficient verified market data\.\s*)?/i, '') || 'No reliable pricing data for this vehicle.'}
            </span>
          </div>
        )}

        {/* Toggle Extended Specifications Accordion */}
        <div style={styles.accordionToggle}>
          <button
            onClick={() => setShowExtended(!showExtended)}
            style={styles.accordionBtn}
            className="interactive-target"
          >
            <span>{showExtended ? 'Collapse Extended Specifications' : 'View Extended Multi-Branch Intelligence'}</span>
            {showExtended ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>

        {/* Extended Details Drawer */}
        {showExtended && (
          <div style={styles.extendedDrawer}>
            {/* Engine & Performance Specs */}
            <div style={styles.drawerSection}>
              <div style={styles.drawerSectionHeader}>
                <Fuel size={14} color="var(--accent-petrol-teal)" />
                <span style={styles.drawerSectionTitle}>POWERTRAIN & SPECS</span>
              </div>
              <div style={styles.drawerGrid}>
                <div style={styles.drawerItem}>
                  <span style={styles.drawerKey}>Horsepower:</span>
                  <span style={styles.drawerVal}>{engine.horsepower ? `${engine.horsepower} HP` : 'Unavailable'}</span>
                </div>
                <div style={styles.drawerItem}>
                  <span style={styles.drawerKey}>Torque:</span>
                  <span style={styles.drawerVal}>{engine.torque_nm ? `${engine.torque_nm} Nm` : 'Unavailable'}</span>
                </div>
                <div style={styles.drawerItem}>
                  <span style={styles.drawerKey}>Fuel:</span>
                  <span style={styles.drawerVal}>{engine.fuel_type || 'Unavailable'}</span>
                </div>
                <div style={styles.drawerItem}>
                  <span style={styles.drawerKey}>Drivetrain:</span>
                  <span style={styles.drawerVal}>{specs.drivetrain || 'Unavailable'}</span>
                </div>
              </div>
            </div>

            {/* Exterior Intelligence (YOLO11n-seg) */}
            {exterior.detected_classes && exterior.detected_classes.length > 0 && (
              <div style={styles.drawerSection}>
                <div style={styles.drawerSectionHeader}>
                  <Layers size={14} color="var(--accent-tangerine)" />
                  <span style={styles.drawerSectionTitle}>EXTERIOR COMPONENTS (YOLO11n-seg)</span>
                </div>
                <div style={styles.tagWrap}>
                  {exterior.detected_classes.slice(0, 12).map((comp, i) => (
                    <span key={i} style={styles.componentTag}>
                      {typeof comp === 'string' ? comp : comp.class_name || comp.name || `Part ${i}`}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Image Quality Score */}
            {typeof quality.overall_score === 'number' && (
              <div style={styles.qualityRow}>
                <span style={styles.qualityLabel}>Input Image Quality Score:</span>
                <span style={styles.qualityScoreVal}>{quality.overall_score.toFixed(1)} / 100</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    width: '100%',
    maxWidth: '420px',
  },
  card: {
    backgroundColor: 'rgba(222, 213, 197, 0.45)',
    backdropFilter: 'blur(10px)',
    border: '1px solid var(--border-subtle)',
    borderRadius: 'var(--radius-lg)',
    padding: '28px 24px',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 12px 32px rgba(32, 32, 30, 0.04)',
    position: 'relative',
  },
  emptyCard: {
    backgroundColor: 'rgba(222, 213, 197, 0.35)',
    backdropFilter: 'blur(8px)',
    border: '1.5px dashed var(--border-dashed)',
    borderRadius: 'var(--radius-lg)',
    padding: '32px 24px',
    display: 'flex',
    flexDirection: 'column',
  },
  headerRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '16px',
  },
  subHeader: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.68rem',
    letterSpacing: '0.2em',
    color: 'var(--text-ink-muted)',
    fontWeight: '700',
  },
  standbyBadge: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.64rem',
    letterSpacing: '0.14em',
    color: 'var(--text-ink-faint)',
    border: '1px solid var(--border-subtle)',
    padding: '2px 8px',
    borderRadius: 'var(--radius-full)',
  },
  verifiedStatusBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.65rem',
    letterSpacing: '0.1em',
    color: 'var(--accent-petrol-teal)',
    backgroundColor: 'var(--accent-petrol-teal-subtle)',
    border: '1px solid var(--accent-petrol-teal-border)',
    padding: '2px 8px',
    borderRadius: 'var(--radius-full)',
    fontWeight: '700',
  },
  emptyBody: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
    padding: '16px 8px',
    gap: '12px',
  },
  emptyIconCircle: {
    width: '54px',
    height: '54px',
    borderRadius: '50%',
    backgroundColor: 'rgba(23, 107, 112, 0.08)',
    border: '1px solid var(--accent-petrol-teal-border)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: '4px',
  },
  emptyTitle: {
    fontFamily: 'var(--font-editorial)',
    fontSize: '1.85rem',
    fontStyle: 'italic',
    fontWeight: '700',
    color: 'var(--text-ink-black)',
    margin: 0,
  },
  emptyDesc: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.86rem',
    lineHeight: 1.5,
    color: 'var(--text-ink-muted)',
    maxWidth: '320px',
    margin: 0,
  },
  standbyBranchList: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    marginTop: '12px',
    padding: '12px 14px',
    backgroundColor: 'rgba(241, 235, 221, 0.6)',
    borderRadius: 'var(--radius-md)',
    border: '1px solid var(--border-subtle)',
  },
  standbyBranchItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    color: 'var(--text-ink-muted)',
    textAlign: 'left',
  },
  standbyBranchDot: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    backgroundColor: 'var(--accent-petrol-teal)',
    opacity: 0.6,
  },
  emptyHint: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.68rem',
    color: 'var(--text-ink-faint)',
    marginTop: '8px',
  },
  loadingBody: {
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
    padding: '8px 0',
  },
  loadingTitle: {
    fontFamily: 'var(--font-editorial)',
    fontSize: '1.75rem',
    fontWeight: '700',
    color: 'var(--text-ink-black)',
    margin: 0,
  },
  loadingDesc: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.85rem',
    color: 'var(--text-ink-muted)',
    lineHeight: 1.5,
    margin: 0,
  },
  loadingPulseTag: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.66rem',
    color: 'var(--accent-tangerine)',
    fontWeight: '700',
    letterSpacing: '0.12em',
  },
  pulseDot: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    backgroundColor: 'var(--accent-tangerine)',
    boxShadow: '0 0 8px var(--accent-tangerine)',
  },
  branchStatusList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    backgroundColor: 'rgba(241, 235, 221, 0.5)',
    padding: '12px 14px',
    borderRadius: 'var(--radius-md)',
    border: '1px solid var(--border-subtle)',
  },
  branchStatusItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    color: 'var(--text-ink-black)',
  },
  branchStatusLabel: {
    flex: 1,
  },
  branchStatusState: {
    color: 'var(--accent-petrol-teal)',
    fontWeight: '700',
  },
  loadingProgressBar: {
    width: '100%',
    height: '4px',
    backgroundColor: 'var(--border-subtle)',
    borderRadius: 'var(--radius-full)',
    overflow: 'hidden',
    marginTop: '8px',
  },
  loadingProgressFill: {
    width: '60%',
    height: '100%',
    background: 'linear-gradient(90deg, var(--accent-petrol-teal), var(--accent-tangerine))',
    animation: 'pulseSubtle 1.2s infinite ease-in-out',
  },
  brandRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  makeTitle: {
    fontFamily: 'var(--font-editorial)',
    fontSize: 'clamp(2.4rem, 4vw, 3.4rem)',
    fontWeight: '700',
    letterSpacing: '-0.02em',
    color: 'var(--text-ink-black)',
    margin: 0,
    lineHeight: 1,
  },
  brandLogoCircle: {
    width: '44px',
    height: '44px',
    borderRadius: 'var(--radius-md)',
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    border: '1px solid var(--border-subtle)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'var(--text-ink-black)',
  },
  confidenceSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    marginBottom: '20px',
  },
  confidenceTag: {
    alignSelf: 'flex-start',
  },
  percentageRow: {
    lineHeight: 1,
  },
  percentageText: {
    fontFamily: 'var(--font-editorial)',
    fontSize: 'clamp(2.6rem, 4.2vw, 3.8rem)',
    fontStyle: 'italic',
    fontWeight: '700',
    color: 'var(--accent-petrol-teal)',
    letterSpacing: '-0.02em',
    lineHeight: 0.95,
  },
  barTrack: {
    width: '100%',
    height: '5px',
    backgroundColor: 'rgba(32, 32, 30, 0.1)',
    borderRadius: 'var(--radius-full)',
    overflow: 'hidden',
    marginTop: '8px',
  },
  barFill: {
    height: '100%',
    backgroundColor: 'var(--accent-petrol-teal)',
    borderRadius: 'var(--radius-full)',
    transition: 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr auto 1fr auto 1fr',
    alignItems: 'center',
    padding: '16px 0',
    borderTop: '1px solid var(--border-subtle)',
    borderBottom: '1px solid var(--border-subtle)',
    marginBottom: '18px',
  },
  metricItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  metricLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.64rem',
    color: 'var(--text-ink-muted)',
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
  },
  metricValue: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.92rem',
    fontWeight: '700',
    color: 'var(--text-ink-black)',
  },
  metricDivider: {
    width: '1px',
    height: '24px',
    backgroundColor: 'var(--border-subtle)',
  },
  topPredictionsContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    marginBottom: '16px',
  },
  topPredHeader: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.66rem',
    letterSpacing: '0.18em',
    color: 'var(--text-ink-muted)',
    fontWeight: '700',
  },
  predList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  predRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontSize: '0.82rem',
  },
  predName: {
    width: '75px',
    fontFamily: 'var(--font-sans)',
    fontWeight: '600',
    color: 'var(--text-ink-black)',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  predBarTrack: {
    flex: 1,
    height: '4px',
    backgroundColor: 'rgba(32, 32, 30, 0.08)',
    borderRadius: 'var(--radius-full)',
    overflow: 'hidden',
  },
  predBarFill: {
    height: '100%',
    borderRadius: 'var(--radius-full)',
    transition: 'width 0.4s ease',
  },
  predPct: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    color: 'var(--text-ink-muted)',
    minWidth: '40px',
    textAlign: 'right',
  },
  valuationBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    backgroundColor: 'rgba(23, 107, 112, 0.08)',
    border: '1px solid var(--accent-petrol-teal-border)',
    padding: '8px 12px',
    borderRadius: 'var(--radius-md)',
    marginBottom: '16px',
  },
  valuationText: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.8rem',
    color: 'var(--text-ink-black)',
  },
  accordionToggle: {
    width: '100%',
  },
  accordionBtn: {
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    background: 'none',
    border: 'none',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    fontWeight: '600',
    letterSpacing: '0.06em',
    color: 'var(--accent-petrol-teal)',
    padding: '8px 0',
    cursor: 'pointer',
  },
  extendedDrawer: {
    marginTop: '12px',
    paddingTop: '12px',
    borderTop: '1px solid var(--border-subtle)',
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
  },
  drawerSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  drawerSectionHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  drawerSectionTitle: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.66rem',
    letterSpacing: '0.12em',
    color: 'var(--text-ink-muted)',
    fontWeight: '700',
  },
  drawerGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '6px',
    backgroundColor: 'rgba(255, 255, 255, 0.4)',
    padding: '8px 10px',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--border-subtle)',
  },
  drawerItem: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '0.75rem',
  },
  drawerKey: {
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-ink-muted)',
  },
  drawerVal: {
    fontFamily: 'var(--font-sans)',
    fontWeight: '600',
    color: 'var(--text-ink-black)',
  },
  tagWrap: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
  },
  componentTag: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.68rem',
    padding: '3px 8px',
    borderRadius: 'var(--radius-full)',
    backgroundColor: 'rgba(255, 255, 255, 0.6)',
    border: '1px solid var(--border-subtle)',
    color: 'var(--text-ink-black)',
  },
  qualityRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '6px 10px',
    backgroundColor: 'rgba(23, 107, 112, 0.06)',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--accent-petrol-teal-border)',
  },
  qualityLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.7rem',
    color: 'var(--text-ink-muted)',
  },
  qualityScoreVal: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.74rem',
    fontWeight: '700',
    color: 'var(--accent-petrol-teal)',
  },
};

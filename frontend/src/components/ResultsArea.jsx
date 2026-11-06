import React, { useState } from 'react';
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

export default function ResultsArea({ resultData, isLoading }) {
  const [showExtended, setShowExtended] = useState(false);

  // ============================================================
  // 1. LOADING STATE
  // ============================================================
  if (isLoading) {
    return (
      <div style={styles.container}>
        <div style={styles.card}>
          <div style={styles.headerRow}>
            <span style={styles.subHeader}>MULTI-BRANCH INFERENCE</span>
            <span style={styles.loadingPulseTag}>
              <span style={styles.pulseDot} />
              ANALYZING
            </span>
          </div>

          <div style={styles.loadingBody}>
            <h3 style={styles.loadingTitle}>Processing Visual Perception</h3>
            <p style={styles.loadingDesc}>
              The three-branch neural pipeline is executing across vehicle identification,
              exterior component segmentation, and Groq multimodal reasoning.
            </p>

            <div style={styles.branchStatusList}>
              <div style={styles.branchStatusItem}>
                <Cpu size={16} color="var(--accent-petrol-teal)" />
                <span style={styles.branchStatusLabel}>Branch 1: Vehicle Feature Classification</span>
                <span style={styles.branchStatusState}>Active</span>
              </div>
              <div style={styles.branchStatusItem}>
                <Eye size={16} color="var(--accent-tangerine)" />
                <span style={styles.branchStatusLabel}>Branch 2: YOLO11n-seg Component Parsing</span>
                <span style={styles.branchStatusState}>Active</span>
              </div>
              <div style={styles.branchStatusItem}>
                <Zap size={16} color="var(--accent-petrol-teal)" />
                <span style={styles.branchStatusLabel}>Branch 3: Groq Multimodal Vision Synthesis</span>
                <span style={styles.branchStatusState}>Active</span>
              </div>
            </div>

            <div style={styles.loadingProgressBar}>
              <div style={styles.loadingProgressFill} />
            </div>
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
  const exterior = resultData.exterior_intelligence || {};
  const quality = resultData.quality_assessment || {};
  const originalB1 = resultData.original_model_outputs?.branch_1_vehicle_id || {};
  const topCandidates = originalB1?.predictions?.[0]?.top5 || [];

  const displayMake = vehicle.make || specs.make || 'Unidentified Vehicle';
  const displayModel = specs.variant || vehicle.model || 'Unknown Model';
  const displayConfidence = vehicle.confidence !== undefined ? vehicle.confidence : null;
  const confidencePercent = displayConfidence !== null ? (displayConfidence * 100).toFixed(1) : null;

  const displayType = engine.configuration || specs.body_type || 'N/A';
  const displayDisplacement = engine.displacement || 'N/A';

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
              <span>{specs.verification_status.toUpperCase()}</span>
            </span>
          )}
        </div>

        {/* Brand Name & Logo Badge */}
        <div style={styles.brandRow}>
          <h2 style={styles.makeTitle}>{displayMake}</h2>
          <div style={styles.brandLogoCircle}>
            {renderBrandBadge(displayMake)}
          </div>
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

        {/* Spec Metrics Row */}
        <div style={styles.metricsGrid}>
          <div style={styles.metricItem}>
            <span style={styles.metricLabel}>Model / Trim</span>
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
            <span style={styles.valuationText}>Estimated Market Value: <strong>{formattedPrice}</strong></span>
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
                  <span style={styles.drawerVal}>{engine.horsepower_hp ? `${engine.horsepower_hp} HP` : 'N/A'}</span>
                </div>
                <div style={styles.drawerItem}>
                  <span style={styles.drawerKey}>Torque:</span>
                  <span style={styles.drawerVal}>{engine.torque_nm ? `${engine.torque_nm} Nm` : 'N/A'}</span>
                </div>
                <div style={styles.drawerItem}>
                  <span style={styles.drawerKey}>Fuel:</span>
                  <span style={styles.drawerVal}>{engine.fuel_type || 'N/A'}</span>
                </div>
                <div style={styles.drawerItem}>
                  <span style={styles.drawerKey}>Drivetrain:</span>
                  <span style={styles.drawerVal}>{specs.transmission?.drivetrain || 'N/A'}</span>
                </div>
              </div>
            </div>

            {/* Exterior Intelligence (YOLO11n-seg) */}
            {exterior.components_detected && (
              <div style={styles.drawerSection}>
                <div style={styles.drawerSectionHeader}>
                  <Layers size={14} color="var(--accent-tangerine)" />
                  <span style={styles.drawerSectionTitle}>EXTERIOR COMPONENTS (YOLO11n-seg)</span>
                </div>
                <div style={styles.tagWrap}>
                  {exterior.components_detected.slice(0, 8).map((comp, i) => (
                    <span key={i} style={styles.componentTag}>
                      {typeof comp === 'string' ? comp : comp.class_name || comp.name || `Part ${i}`}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Image Quality Score */}
            {quality.overall_quality_score !== undefined && (
              <div style={styles.qualityRow}>
                <span style={styles.qualityLabel}>Input Image Quality Score:</span>
                <span style={styles.qualityScoreVal}>{quality.overall_quality_score.toFixed(1)} / 100</span>
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

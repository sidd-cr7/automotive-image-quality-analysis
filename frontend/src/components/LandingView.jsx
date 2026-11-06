import React from 'react';
import { ArrowUpRight, Cpu, Eye, ShieldCheck, Zap, Layers, Sparkles } from 'lucide-react';
import SurrealCollage from './SurrealCollage';

export default function LandingView({ onNavigateToDetect }) {
  return (
    <div style={styles.container}>
      {/* ============================================================
          HERO SECTION (Full viewport editorial experience)
          ============================================================ */}
      <section style={styles.heroSection}>
        {/* Top Meta Bar */}
        <div style={styles.topMeta}>
          <span style={styles.metaLabel}>
            AUTOMOTIVE COMPUTER VISION <span style={styles.metaDash}>—</span>
          </span>
        </div>

        {/* Big Editorial Title with interactive presence */}
        <div style={styles.titleWrapper}>
          <h1 style={styles.titleLine1}>ENGINE</h1>
          <h1 style={styles.titleLine2}>INTELLIGENCE</h1>
        </div>

        {/* Subtitle Triad */}
        <div style={styles.subHeader}>
          <span style={styles.subItem}>DETECT</span>
          <span style={styles.subDivider}>/</span>
          <span style={styles.subItem}>IDENTIFY</span>
          <span style={styles.subDivider}>/</span>
          <span style={styles.subItem}>ANALYZE</span>
        </div>

        {/* Expansive Surrealist Artwork Hero Feature */}
        <div style={styles.artworkHeroContainer}>
          <SurrealCollage mode="hero" />
        </div>

        {/* Editorial Hero Intro & Prominent CTA to Detection Workspace */}
        <div style={styles.heroCtaRow}>
          <p style={styles.heroDescription}>
            A synthesis of classical aesthetic form and multi-model neural perception.
            Autonomous vehicle identification, exterior segmentation, interior inspection,
            and engine intelligence operating across three synchronized deep vision branches.
          </p>

          <button
            onClick={onNavigateToDetect}
            className="btn-editorial-teal interactive-target"
            style={styles.launchBtn}
          >
            <span>ENTER DETECTION WORKSPACE</span>
            <ArrowUpRight size={18} strokeWidth={2.4} />
          </button>
        </div>
      </section>

      {/* ============================================================
          EDITORIAL SECTION 1: THREE-BRANCH NEURAL ARCHITECTURE
          ============================================================ */}
      <section style={styles.manifestoSection}>
        <div style={styles.sectionHeader}>
          <span style={styles.sectionTag}>FIG. 02 — ARCHITECTURAL DEPLOYMENT</span>
          <h2 style={styles.sectionTitle}>The Three-Branch Neural Perception Engine</h2>
          <p style={styles.sectionSubtitle}>
            Engineered to overcome the limitations of monolithic models by deploying
            domain-specialized architectures in parallel.
          </p>
        </div>

        <div style={styles.branchesGrid}>
          {/* Branch 1 Card */}
          <div style={styles.branchCard}>
            <div style={styles.branchCardHeader}>
              <div style={styles.branchBadgeTeal}>BRANCH 01</div>
              <Cpu size={20} color="var(--accent-petrol-teal)" />
            </div>
            <h3 style={styles.branchTitle}>Vehicle Identification</h3>
            <p style={styles.branchDesc}>
              YOLOv8 vehicle bounding-box detection fused with ResNet-50 deep feature
              extraction. Predicts make, model, and generation with top-5 candidate rankings
              and predictive confidence scoring.
            </p>
            <div style={styles.branchMeta}>
              <span>YOLOv8 + ResNet-50</span>
              <span style={styles.metaDot}>•</span>
              <span>1000+ Vehicle Classes</span>
            </div>
          </div>

          {/* Branch 2 Card */}
          <div style={styles.branchCard}>
            <div style={styles.branchCardHeader}>
              <div style={styles.branchBadgeTangerine}>BRANCH 02</div>
              <Eye size={20} color="var(--accent-tangerine)" />
            </div>
            <h3 style={styles.branchTitle}>Exterior Intelligence</h3>
            <p style={styles.branchDesc}>
              Custom-trained YOLO11n-seg instance segmentation across 23 exterior component
              classes (bumpers, hood, fenders, grilles, lights). Segments boundaries and flags
              aftermarket modifications.
            </p>
            <div style={styles.branchMeta}>
              <span>YOLO11n-seg</span>
              <span style={styles.metaDot}>•</span>
              <span>23 Component Classes</span>
            </div>
          </div>

          {/* Branch 3 Card */}
          <div style={styles.branchCard}>
            <div style={styles.branchCardHeader}>
              <div style={styles.branchBadgeTeal}>BRANCH 03</div>
              <Zap size={20} color="var(--accent-petrol-teal)" />
            </div>
            <h3 style={styles.branchTitle}>Interior & Engine Reasoning</h3>
            <p style={styles.branchDesc}>
              Groq Llama-Vision multimodal cognition with automated primary-to-backup API
              failover. Diagnoses engine bays, manifolds, intake assemblies, steering wheels,
              and cabin cockpit ergonomics.
            </p>
            <div style={styles.branchMeta}>
              <span>Dual Groq Vision</span>
              <span style={styles.metaDot}>•</span>
              <span>Sub-second Latency</span>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          EDITORIAL SECTION 2: ADAPTIVE CV RESTORATION & SPECIFICATIONS
          ============================================================ */}
      <section style={styles.specificationsSection}>
        <div style={styles.twoColumnGrid}>
          {/* Left Column: Image Quality & Enhancement */}
          <div style={styles.specCard}>
            <div style={styles.specHeaderRow}>
              <Layers size={22} color="var(--accent-petrol-teal)" />
              <span style={styles.specTag}>PART 2 & 3 PIPELINE</span>
            </div>
            <h3 style={styles.specCardTitle}>Intelligent Quality Assessment & Adaptive CV</h3>
            <p style={styles.specCardBody}>
              Every frame undergoes rigorous defect quantification before inference.
              Detects underexposure, overexposure, motion blur, sensor noise, and specular
              glare. Automatically applies CLAHE contrast optimization, unsharp masking,
              and comparative entropy selection between original and enhanced images.
            </p>
            <div style={styles.pillList}>
              <span style={styles.techPill}>CLAHE Lighting Balance</span>
              <span style={styles.techPill}>Laplacian Blur Detection</span>
              <span style={styles.techPill}>Dual-Image Routing</span>
            </div>
          </div>

          {/* Right Column: Specifications & Market Pricing */}
          <div style={styles.specCard}>
            <div style={styles.specHeaderRow}>
              <ShieldCheck size={22} color="var(--accent-tangerine)" />
              <span style={styles.specTag}>PART 4 INTELLIGENCE</span>
            </div>
            <h3 style={styles.specCardTitle}>Verified Specifications & Current Valuation</h3>
            <p style={styles.specCardBody}>
              Transforms raw model predictions into actionable automotive intelligence.
              Retrieves authentic manufacturer engine configurations (displacement, horsepower,
              torque, cylinders, transmission) and computes market valuations across regional
              currencies.
            </p>
            <div style={styles.pillList}>
              <span style={styles.techPill}>Verified Engine Specs</span>
              <span style={styles.techPill}>Market Valuation Engine</span>
              <span style={styles.techPill}>Regional Currencies</span>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          FINAL CALL TO ACTION BANNER
          ============================================================ */}
      <section style={styles.bottomCtaSection}>
        <div style={styles.bottomCtaCard}>
          <span style={styles.bottomCtaTag}>READY FOR INFERENCE</span>
          <h2 style={styles.bottomCtaTitle}>Experience Multi-Branch Automotive Perception</h2>
          <p style={styles.bottomCtaText}>
            Test your engine bays, exterior profiles, and vehicle photographs against
            the live computer vision backend.
          </p>
          <button
            onClick={onNavigateToDetect}
            className="btn-editorial-teal interactive-target"
            style={styles.bottomCtaBtn}
          >
            <span>LAUNCH DETECTION LABORATORY</span>
            <ArrowUpRight size={18} strokeWidth={2.4} />
          </button>
        </div>
      </section>
    </div>
  );
}

const styles = {
  container: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    gap: '80px',
    paddingBottom: '80px',
  },
  heroSection: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    paddingTop: '8px',
  },
  topMeta: {
    display: 'flex',
    justifyContent: 'flex-end',
    width: '100%',
    paddingBottom: '8px',
  },
  metaLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.74rem',
    letterSpacing: '0.24em',
    color: 'var(--text-ink-black)',
    fontWeight: '700',
    opacity: 0.88,
  },
  metaDash: {
    color: 'var(--accent-petrol-teal)',
    fontWeight: '700',
  },
  titleWrapper: {
    lineHeight: 0.88,
    userSelect: 'none',
  },
  titleLine1: {
    fontFamily: 'var(--font-editorial)',
    fontSize: 'clamp(3.8rem, 8.2vw, 7.8rem)',
    fontStyle: 'italic',
    fontWeight: '700',
    letterSpacing: '-0.025em',
    color: 'var(--text-ink-black)',
    margin: 0,
    textTransform: 'uppercase',
  },
  titleLine2: {
    fontFamily: 'var(--font-editorial)',
    fontSize: 'clamp(3.8rem, 8.2vw, 7.8rem)',
    fontStyle: 'italic',
    fontWeight: '700',
    letterSpacing: '-0.025em',
    color: 'var(--text-ink-black)',
    margin: 0,
    textTransform: 'uppercase',
  },
  subHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '14px',
    marginTop: '16px',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.76rem',
    letterSpacing: '0.28em',
    color: 'var(--text-ink-black)',
    opacity: 0.85,
    fontWeight: '600',
  },
  subItem: {
    letterSpacing: '0.24em',
  },
  subDivider: {
    color: 'var(--accent-petrol-teal)',
    opacity: 0.7,
  },
  artworkHeroContainer: {
    width: '100%',
    position: 'relative',
    margin: '24px 0 32px 0',
  },
  heroCtaRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '40px',
    backgroundColor: 'rgba(222, 213, 197, 0.45)',
    border: '1px solid var(--border-subtle)',
    padding: '24px 32px',
    borderRadius: 'var(--radius-lg)',
    flexWrap: 'wrap',
  },
  heroDescription: {
    flex: '1 1 450px',
    fontFamily: 'var(--font-sans)',
    fontSize: '1.02rem',
    lineHeight: 1.6,
    color: 'var(--text-ink-muted)',
    margin: 0,
  },
  launchBtn: {
    flexShrink: 0,
    fontSize: '0.96rem',
    padding: '14px 28px',
  },
  manifestoSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '36px',
  },
  sectionHeader: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  sectionTag: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    letterSpacing: '0.2em',
    color: 'var(--accent-petrol-teal)',
    fontWeight: '700',
  },
  sectionTitle: {
    fontFamily: 'var(--font-editorial)',
    fontSize: 'clamp(2.2rem, 4vw, 3.2rem)',
    fontWeight: '700',
    color: 'var(--text-ink-black)',
    letterSpacing: '-0.02em',
    margin: 0,
  },
  sectionSubtitle: {
    fontFamily: 'var(--font-sans)',
    fontSize: '1rem',
    color: 'var(--text-ink-muted)',
    maxWidth: '700px',
    lineHeight: 1.5,
  },
  branchesGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '24px',
  },
  branchCard: {
    backgroundColor: 'rgba(222, 213, 197, 0.35)',
    border: '1px solid var(--border-subtle)',
    borderRadius: 'var(--radius-lg)',
    padding: '28px 24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
    transition: 'transform 0.2s ease, border-color 0.2s ease',
  },
  branchCardHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  branchBadgeTeal: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.68rem',
    fontWeight: '700',
    letterSpacing: '0.12em',
    padding: '4px 10px',
    borderRadius: 'var(--radius-full)',
    backgroundColor: 'var(--accent-petrol-teal-subtle)',
    color: 'var(--accent-petrol-teal)',
    border: '1px solid var(--accent-petrol-teal-border)',
  },
  branchBadgeTangerine: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.68rem',
    fontWeight: '700',
    letterSpacing: '0.12em',
    padding: '4px 10px',
    borderRadius: 'var(--radius-full)',
    backgroundColor: 'var(--accent-tangerine-subtle)',
    color: 'var(--accent-tangerine)',
    border: '1px solid var(--accent-tangerine-border)',
  },
  branchTitle: {
    fontFamily: 'var(--font-editorial)',
    fontSize: '1.65rem',
    fontWeight: '700',
    color: 'var(--text-ink-black)',
    margin: 0,
  },
  branchDesc: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.88rem',
    lineHeight: 1.55,
    color: 'var(--text-ink-muted)',
    margin: 0,
    flex: 1,
  },
  branchMeta: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    color: 'var(--text-ink-faint)',
    borderTop: '1px solid var(--border-subtle)',
    paddingTop: '12px',
    marginTop: '6px',
  },
  metaDot: {
    color: 'var(--accent-petrol-teal)',
  },
  specificationsSection: {
    width: '100%',
  },
  twoColumnGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
    gap: '24px',
  },
  specCard: {
    backgroundColor: 'rgba(222, 213, 197, 0.45)',
    border: '1px solid var(--border-subtle)',
    borderRadius: 'var(--radius-lg)',
    padding: '32px 28px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  specHeaderRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  specTag: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    fontWeight: '700',
    letterSpacing: '0.16em',
    color: 'var(--text-ink-muted)',
  },
  specCardTitle: {
    fontFamily: 'var(--font-editorial)',
    fontSize: '1.75rem',
    fontWeight: '700',
    color: 'var(--text-ink-black)',
    margin: 0,
  },
  specCardBody: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.92rem',
    lineHeight: 1.6,
    color: 'var(--text-ink-muted)',
    margin: 0,
  },
  pillList: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    marginTop: '8px',
  },
  techPill: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    padding: '4px 10px',
    borderRadius: 'var(--radius-full)',
    backgroundColor: 'rgba(255, 255, 255, 0.6)',
    border: '1px solid var(--border-subtle)',
    color: 'var(--text-ink-black)',
  },
  bottomCtaSection: {
    width: '100%',
  },
  bottomCtaCard: {
    backgroundColor: 'var(--bg-charcoal)',
    color: 'var(--text-light-bright)',
    padding: '52px 48px',
    borderRadius: 'var(--radius-xl)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
    gap: '18px',
    position: 'relative',
    overflow: 'hidden',
    boxShadow: '0 24px 48px rgba(21, 21, 19, 0.35)',
  },
  bottomCtaTag: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.74rem',
    letterSpacing: '0.25em',
    color: 'var(--accent-tangerine)',
    fontWeight: '700',
  },
  bottomCtaTitle: {
    fontFamily: 'var(--font-editorial)',
    fontSize: 'clamp(2rem, 3.8vw, 3rem)',
    fontWeight: '700',
    margin: 0,
  },
  bottomCtaText: {
    fontFamily: 'var(--font-sans)',
    fontSize: '1rem',
    color: 'var(--text-light-muted)',
    maxWidth: '560px',
    margin: '0 0 12px 0',
  },
  bottomCtaBtn: {
    fontSize: '0.96rem',
    padding: '14px 32px',
  },
};

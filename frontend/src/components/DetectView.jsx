import React from 'react';
import { ArrowLeft, Sparkles, Terminal } from 'lucide-react';
import UploadArea from './UploadArea';
import SurrealCollage from './SurrealCollage';
import ResultsArea from './ResultsArea';

export default function DetectView({
  onNavigateToLanding,
  onImageSelected,
  onAnalyze,
  isAnalyzing,
  selectedFile,
  imagePreview,
  onReset,
  errorMessage,
  resultData,
}) {
  return (
    <div style={styles.container}>
      {/* Top Workspace Navigation Bar */}
      <div style={styles.topBar}>
        <button
          onClick={onNavigateToLanding}
          style={styles.backBtn}
          className="interactive-target"
        >
          <ArrowLeft size={16} />
          <span>Return to Editorial Story</span>
        </button>

        <div style={styles.statusIndicator}>
          <div style={styles.statusDot} />
          <span>LABORATORY ACTIVE — 3 BRANCHES ONLINE</span>
        </div>
      </div>

      {/* Workspace Header */}
      <header style={styles.header}>
        <div style={styles.headerMeta}>
          <span style={styles.metaLabel}>WORKSPACE 01 — MULTI-MODEL PERCEPTION</span>
        </div>
        <h1 style={styles.title}>Visual Inference Laboratory</h1>
        <p style={styles.subtitle}>
          Upload high-resolution vehicle exteriors, engine bays, or interiors. The system
          executes intelligent image assessment, adaptive CV enhancement, and multi-branch
          neural analysis concurrently.
        </p>
      </header>

      {/* 3-Column Asymmetric Detection Grid */}
      <div style={styles.workspaceGrid}>
        {/* Column 1: Exactly ONE Upload Area */}
        <section style={styles.leftCol}>
          <UploadArea
            onImageSelected={onImageSelected}
            onAnalyze={onAnalyze}
            isAnalyzing={isAnalyzing}
            selectedFile={selectedFile}
            imagePreview={imagePreview}
            onReset={onReset}
            errorMessage={errorMessage}
          />
        </section>

        {/* Column 2: Center Surrealist Visual Collage (with live scan glow during inference) */}
        <section style={styles.centerCol}>
          <SurrealCollage mode="compact" isAnalyzing={isAnalyzing} />
        </section>

        {/* Column 3: Results Area (NO FABRICATED RESULTS) */}
        <section style={styles.rightCol}>
          <ResultsArea resultData={resultData} isLoading={isAnalyzing} />
        </section>
      </div>
    </div>
  );
}

const styles = {
  container: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
    paddingBottom: '60px',
  },
  topBar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: '16px',
    borderBottom: '1px solid var(--border-subtle)',
    flexWrap: 'wrap',
    gap: '12px',
  },
  backBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '8px',
    background: 'none',
    border: 'none',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.78rem',
    color: 'var(--accent-petrol-teal)',
    cursor: 'pointer',
    fontWeight: '600',
    padding: '4px 0',
  },
  statusIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.68rem',
    color: 'var(--text-ink-muted)',
    letterSpacing: '0.12em',
  },
  statusDot: {
    width: '7px',
    height: '7px',
    borderRadius: '50%',
    backgroundColor: 'var(--accent-petrol-teal)',
    boxShadow: '0 0 6px var(--accent-petrol-teal)',
  },
  header: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  headerMeta: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.7rem',
    letterSpacing: '0.2em',
    color: 'var(--accent-tangerine)',
    fontWeight: '700',
  },
  title: {
    fontFamily: 'var(--font-editorial)',
    fontSize: 'clamp(2.4rem, 4.5vw, 3.6rem)',
    fontWeight: '700',
    color: 'var(--text-ink-black)',
    letterSpacing: '-0.02em',
    margin: 0,
    lineHeight: 1.05,
  },
  subtitle: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.94rem',
    lineHeight: 1.5,
    color: 'var(--text-ink-muted)',
    maxWidth: '720px',
    margin: 0,
  },
  workspaceGrid: {
    display: 'grid',
    gridTemplateColumns: 'minmax(300px, 360px) 1fr minmax(320px, 420px)',
    alignItems: 'center',
    gap: '28px',
    marginTop: '16px',
    position: 'relative',
    width: '100%',
  },
  leftCol: {
    display: 'flex',
    justifyContent: 'flex-start',
    zIndex: 10,
  },
  centerCol: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 5,
  },
  rightCol: {
    display: 'flex',
    justifyContent: 'flex-end',
    zIndex: 10,
  },
};

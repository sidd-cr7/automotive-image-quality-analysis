import React from 'react';

export default function Hero() {
  return (
    <header style={styles.header}>
      {/* Top Meta Tagline */}
      <div style={styles.topMeta}>
        <span style={styles.metaLabel}>
          AUTOMOTIVE COMPUTER VISION <span style={styles.metaDash}>—</span>
        </span>
      </div>

      {/* Main Editorial Title */}
      <div style={styles.titleWrapper}>
        <h1 style={styles.titleLine1}>ENGINE</h1>
        <h1 style={styles.titleLine2}>INTELLIGENCE</h1>
      </div>

      {/* Triad Subtitle */}
      <div style={styles.subHeader}>
        <span style={styles.subItem}>DETECT</span>
        <span style={styles.subDivider}>/</span>
        <span style={styles.subItem}>IDENTIFY</span>
        <span style={styles.subDivider}>/</span>
        <span style={styles.subItem}>ANALYZE</span>
      </div>
    </header>
  );
}

const styles = {
  header: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    zIndex: 10,
    marginBottom: '20px',
  },
  topMeta: {
    display: 'flex',
    justifyContent: 'flex-end',
    width: '100%',
    paddingBottom: '8px',
  },
  metaLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    letterSpacing: '0.22em',
    color: 'var(--text-ink-black)',
    fontWeight: '700',
    opacity: 0.85,
  },
  metaDash: {
    color: 'var(--accent-terracotta)',
    fontWeight: '700',
  },
  titleWrapper: {
    lineHeight: 0.88,
    userSelect: 'none',
  },
  titleLine1: {
    fontFamily: 'var(--font-editorial)',
    fontSize: 'clamp(3.8rem, 7.5vw, 6.8rem)',
    fontStyle: 'italic',
    fontWeight: '700',
    letterSpacing: '-0.02em',
    color: 'var(--text-ink-black)',
    margin: 0,
    textTransform: 'uppercase',
  },
  titleLine2: {
    fontFamily: 'var(--font-editorial)',
    fontSize: 'clamp(3.8rem, 7.5vw, 6.8rem)',
    fontStyle: 'italic',
    fontWeight: '700',
    letterSpacing: '-0.02em',
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
    fontSize: '0.75rem',
    letterSpacing: '0.28em',
    color: 'var(--text-ink-black)',
    opacity: 0.8,
    fontWeight: '600',
  },
  subItem: {
    letterSpacing: '0.24em',
  },
  subDivider: {
    color: 'var(--accent-terracotta)',
    opacity: 0.6,
  },
};

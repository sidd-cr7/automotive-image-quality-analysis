import React from 'react';

export default function SurrealVisual({ isAnalyzing }) {
  return (
    <div style={styles.container}>
      {/* Central Art Element */}
      <div style={styles.artFrame}>
        <img
          src="/surreal_sculpture_center.jpg"
          alt="Surrealist Automotive Art — Classical Bust with Mechanical Engine Visage and Winding Road"
          style={{
            ...styles.sculptureImg,
            filter: isAnalyzing ? 'contrast(1.08) saturate(1.15)' : 'none',
          }}
          onError={(e) => {
            // Fallback to full reference if cropped image missing
            e.target.src = '/editorial_reference.jpg';
          }}
        />

        {/* Subtle Analysis Scanner Glow when inference is running */}
        {isAnalyzing && (
          <div style={styles.scannerLine}>
            <div style={styles.scannerPulse} />
          </div>
        )}

        {/* Artistic Corner Accents */}
        <div style={styles.cornerLabel}>
          <span style={styles.cornerText}>FIG. 04 / MECHANICAL METAMORPHOSIS</span>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    height: '100%',
    minHeight: '440px',
    overflow: 'visible',
    pointerEvents: 'none',
    userSelect: 'none',
  },
  artFrame: {
    position: 'relative',
    width: '100%',
    maxWidth: '460px',
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  sculptureImg: {
    width: '100%',
    height: 'auto',
    maxHeight: '520px',
    objectFit: 'contain',
    mixBlendMode: 'multiply',
    opacity: 0.95,
    transition: 'filter 0.4s ease',
  },
  scannerLine: {
    position: 'absolute',
    top: '30%',
    left: '10%',
    right: '10%',
    height: '2px',
    background: 'linear-gradient(90deg, transparent, #D4553F, #F28B45, transparent)',
    boxShadow: '0 0 12px rgba(212, 85, 63, 0.8)',
    animation: 'pulseSubtle 1.2s infinite ease-in-out',
  },
  scannerPulse: {
    width: '100%',
    height: '100%',
  },
  cornerLabel: {
    position: 'absolute',
    bottom: '-12px',
    left: '50%',
    transform: 'translateX(-50%)',
    backgroundColor: 'rgba(241, 235, 221, 0.85)',
    padding: '2px 8px',
    borderRadius: '4px',
    border: '1px solid rgba(32, 32, 30, 0.1)',
  },
  cornerText: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.62rem',
    letterSpacing: '0.12em',
    color: 'var(--text-ink-muted)',
    fontWeight: '600',
  },
};

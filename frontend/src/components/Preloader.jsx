import React, { useEffect, useState } from 'react';

export default function Preloader({ onComplete }) {
  const [phase, setPhase] = useState('drawing'); // 'drawing' -> 'revealing' -> 'dissolving' -> 'done'

  useEffect(() => {
    // Check if user already saw the preloader in this session
    const hasVisited = sessionStorage.getItem('ei_intro_seen');
    if (hasVisited) {
      onComplete?.();
      return;
    }

    const t1 = setTimeout(() => setPhase('revealing'), 600);
    const t2 = setTimeout(() => setPhase('dissolving'), 1300);
    const t3 = setTimeout(() => {
      sessionStorage.setItem('ei_intro_seen', 'true');
      setPhase('done');
      onComplete?.();
    }, 1750);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, [onComplete]);

  if (phase === 'done') return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'var(--bg-warm-chalk)',
        zIndex: 999999,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        opacity: phase === 'dissolving' ? 0 : 1,
        transition: 'opacity 0.45s cubic-bezier(0.16, 1, 0.3, 1)',
        pointerEvents: phase === 'dissolving' ? 'none' : 'auto',
      }}
    >
      {/* Editorial wireframe car vector with drawing animation */}
      <div style={{ position: 'relative', width: '160px', height: '60px', marginBottom: '28px' }}>
        <svg
          viewBox="0 0 160 60"
          fill="none"
          stroke="var(--accent-petrol-teal)"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ width: '100%', height: '100%' }}
        >
          {/* Chassis outline */}
          <path
            d="M 10 42 C 20 42 26 34 38 34 H 122 C 134 34 140 42 150 42 M 30 34 L 46 14 H 114 L 130 34 M 35 14 L 125 14"
            strokeDasharray="400"
            strokeDashoffset={phase === 'drawing' ? '200' : '0'}
            style={{
              transition: 'stroke-dashoffset 1.1s cubic-bezier(0.16, 1, 0.3, 1)',
            }}
          />
          {/* Wheels */}
          <circle
            cx="38"
            cy="42"
            r="10"
            stroke="var(--text-ink-black)"
            strokeWidth="2"
            fill="none"
            strokeDasharray="70"
            strokeDashoffset={phase === 'drawing' ? '40' : '0'}
            style={{ transition: 'stroke-dashoffset 0.8s ease 0.2s' }}
          />
          <circle
            cx="122"
            cy="42"
            r="10"
            stroke="var(--text-ink-black)"
            strokeWidth="2"
            fill="none"
            strokeDasharray="70"
            strokeDashoffset={phase === 'drawing' ? '40' : '0'}
            style={{ transition: 'stroke-dashoffset 0.8s ease 0.2s' }}
          />
          {/* Hubcap accents */}
          <circle cx="38" cy="42" r="3" fill="var(--accent-tangerine)" stroke="none" />
          <circle cx="122" cy="42" r="3" fill="var(--accent-tangerine)" stroke="none" />
        </svg>
      </div>

      {/* Title with editorial typographic reveal */}
      <div style={{ textAlign: 'center', overflow: 'hidden' }}>
        <h1
          style={{
            fontFamily: 'var(--font-editorial)',
            fontSize: 'clamp(2.2rem, 4vw, 3.4rem)',
            fontStyle: 'italic',
            fontWeight: '700',
            letterSpacing: '-0.02em',
            color: 'var(--text-ink-black)',
            transform: phase === 'drawing' ? 'translateY(100%)' : 'translateY(0)',
            opacity: phase === 'drawing' ? 0 : 1,
            transition: 'transform 0.6s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.6s ease',
            margin: 0,
            lineHeight: 1,
          }}
        >
          ENGINE INTELLIGENCE
        </h1>

        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.72rem',
            letterSpacing: '0.3em',
            color: 'var(--accent-petrol-teal)',
            marginTop: '12px',
            opacity: phase === 'drawing' ? 0 : 0.9,
            transition: 'opacity 0.6s ease 0.2s',
          }}
        >
          AUTOMOTIVE COMPUTER VISION / 3-BRANCH LAB
        </div>
      </div>
    </div>
  );
}

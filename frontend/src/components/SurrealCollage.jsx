import React, { useEffect, useState, useRef } from 'react';

export default function SurrealCollage({ mode = 'hero', isAnalyzing = false }) {
  const [scrollY, setScrollY] = useState(0);
  const [mouseOffset, setMouseOffset] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };

    const handleMouseMove = (e) => {
      if (window.innerWidth < 768) return;
      const { innerWidth, innerHeight } = window;
      const x = (e.clientX - innerWidth / 2) / (innerWidth / 2);
      const y = (e.clientY - innerHeight / 2) / (innerHeight / 2);
      setMouseOffset({ x, y });
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('mousemove', handleMouseMove, { passive: true });

    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  // Multi-tier parallax offsets
  // Sun & Mountains slowest, Bust medium, Visor & Road faster, Car rides along the curve
  const sunY = scrollY * 0.08 + mouseOffset.y * 8;
  const sunX = mouseOffset.x * 6;

  const mountainY = scrollY * 0.14 + mouseOffset.y * 12;
  const mountainX = mouseOffset.x * 8;

  const bustY = scrollY * 0.26 + mouseOffset.y * 16;
  const bustX = mouseOffset.x * 12;

  const visorY = scrollY * 0.36 + mouseOffset.y * 20;
  const visorX = mouseOffset.x * 15;

  const roadY = scrollY * 0.44 + mouseOffset.y * 22;
  const roadX = mouseOffset.x * 14;

  // Car trajectory along the winding road curve as scroll progress advances
  // Clamped travel progress between 0 and 1
  const maxScroll = typeof window !== 'undefined' ? (document.documentElement.scrollHeight - window.innerHeight || 800) : 800;
  const carProgress = Math.min(1, Math.max(0, scrollY / Math.max(1, maxScroll)));
  // Parametric curve approximating the winding road in the visual
  const carTravelX = carProgress * 90 - mouseOffset.x * 10;
  const carTravelY = carProgress * 45 - mouseOffset.y * 8;
  const carRotation = -4 + carProgress * 8;

  const isHero = mode === 'hero';

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        height: isHero ? 'min(78vh, 680px)' : '420px',
        maxWidth: isHero ? '920px' : '460px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        userSelect: 'none',
        pointerEvents: 'none',
        overflow: 'visible',
      }}
    >
      {/* ============================================================
          LAYER 0: RISING CRIMSON / TANGERINE SUN DISC
          ============================================================ */}
      <div
        style={{
          position: 'absolute',
          top: isHero ? '4%' : '8%',
          right: isHero ? '18%' : '14%',
          width: isHero ? '260px' : '170px',
          height: isHero ? '260px' : '170px',
          borderRadius: '50%',
          background: 'radial-gradient(circle at 45% 45%, #F28B45 0%, #D4553F 65%, #B33825 100%)',
          boxShadow: '0 0 50px rgba(242, 139, 69, 0.45)',
          transform: `translate3d(${sunX}px, ${sunY}px, 0)`,
          transition: 'transform 0.1s ease-out',
          zIndex: 1,
          opacity: 0.94,
        }}
      />

      {/* ============================================================
          LAYER 1: ETCHED MOUNTAINS HORIZON (Slowest Parallax)
          ============================================================ */}
      <div
        style={{
          position: 'absolute',
          top: isHero ? '6%' : '10%',
          right: isHero ? '0%' : '-5%',
          width: isHero ? '72%' : '80%',
          maxWidth: '650px',
          transform: `translate3d(${mountainX}px, ${mountainY}px, 0)`,
          transition: 'transform 0.12s ease-out',
          zIndex: 2,
          mixBlendMode: 'multiply',
          opacity: 0.9,
        }}
      >
        <img
          src="/layer_mountains.png"
          alt="Surrealist Mountain Horizon"
          style={{
            width: '100%',
            height: 'auto',
            display: 'block',
          }}
          onError={(e) => {
            e.target.style.display = 'none';
          }}
        />
      </div>

      {/* ============================================================
          LAYER 2: CLASSICAL MARBLE SCULPTURE BUST (Medium Parallax)
          ============================================================ */}
      <div
        style={{
          position: 'absolute',
          top: isHero ? '18%' : '15%',
          left: isHero ? '28%' : '24%',
          width: isHero ? '42%' : '48%',
          maxWidth: '380px',
          transform: `translate3d(${bustX}px, ${bustY}px, 0)`,
          transition: 'transform 0.14s ease-out',
          zIndex: 3,
          mixBlendMode: 'multiply',
        }}
      >
        <img
          src="/layer_bust.png"
          alt="Classical Marble Sculpture"
          style={{
            width: '100%',
            height: 'auto',
            display: 'block',
            filter: 'contrast(1.05)',
          }}
        />
      </div>

      {/* ============================================================
          LAYER 3: MECHANICAL ENGINE BLOCK VISOR (Faster Parallax)
          ============================================================ */}
      <div
        style={{
          position: 'absolute',
          top: isHero ? '23%' : '21%',
          left: isHero ? '30%' : '26%',
          width: isHero ? '38%' : '44%',
          maxWidth: '340px',
          transform: `translate3d(${visorX}px, ${visorY}px, 0) rotate(-6.5deg)`,
          transition: 'transform 0.16s ease-out',
          zIndex: 5,
          boxShadow: isAnalyzing ? '0 0 24px rgba(242, 139, 69, 0.7)' : '0 12px 28px rgba(32, 32, 30, 0.35)',
        }}
      >
        <img
          src="/layer_visor.png"
          alt="Mechanical Engine Block Visor"
          style={{
            width: '100%',
            height: 'auto',
            display: 'block',
            filter: isAnalyzing ? 'saturate(1.25) contrast(1.1)' : 'contrast(1.04)',
          }}
        />

        {/* Subtle Analysis Scanner Glow when inference is running */}
        {isAnalyzing && (
          <div
            style={{
              position: 'absolute',
              top: '40%',
              left: 0,
              right: 0,
              height: '3px',
              background: 'linear-gradient(90deg, transparent, var(--accent-petrol-teal), var(--accent-tangerine), transparent)',
              boxShadow: '0 0 14px var(--accent-tangerine)',
              animation: 'pulseSubtle 1.2s infinite ease-in-out',
            }}
          />
        )}
      </div>

      {/* ============================================================
          LAYER 4: FLOATING WINDING ROAD & CHECKERED PIER
          ============================================================ */}
      <div
        style={{
          position: 'absolute',
          bottom: isHero ? '4%' : '2%',
          left: isHero ? '12%' : '8%',
          width: isHero ? '58%' : '65%',
          maxWidth: '520px',
          transform: `translate3d(${roadX}px, ${roadY}px, 0)`,
          transition: 'transform 0.18s ease-out',
          zIndex: 6,
          mixBlendMode: 'multiply',
        }}
      >
        <img
          src="/layer_road.png"
          alt="Floating Winding Mountain Highway with Checkered Pier"
          style={{
            width: '100%',
            height: 'auto',
            display: 'block',
          }}
        />
      </div>

      {/* ============================================================
          LAYER 5: MINIATURE VINTAGE VEHICLE TRAVELING THE ROAD
          ============================================================ */}
      <div
        style={{
          position: 'absolute',
          bottom: isHero ? '21%' : '19%',
          left: isHero ? '33%' : '29%',
          width: isHero ? '18%' : '20%',
          maxWidth: '150px',
          transform: `translate3d(${roadX + carTravelX}px, ${roadY + carTravelY}px, 0) rotate(${carRotation}deg)`,
          transition: 'transform 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
          zIndex: 7,
          filter: 'drop-shadow(0 4px 6px rgba(32, 32, 30, 0.35))',
        }}
      >
        <img
          src="/layer_car_clean.png"
          alt="Miniature Vintage Coupe Riding Floating Highway"
          style={{
            width: '100%',
            height: 'auto',
            display: 'block',
          }}
        />
      </div>

      {/* ============================================================
          EDITORIAL FIGURE ACCENT CAPTION
          ============================================================ */}
      <div
        style={{
          position: 'absolute',
          bottom: isHero ? '-14px' : '-8px',
          right: isHero ? '8%' : '4%',
          backgroundColor: 'rgba(241, 235, 221, 0.92)',
          backdropFilter: 'blur(4px)',
          padding: '4px 10px',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-subtle)',
          zIndex: 8,
        }}
      >
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.64rem',
            letterSpacing: '0.14em',
            color: 'var(--text-ink-muted)',
            fontWeight: '600',
          }}
        >
          FIG. 04 / SYNTHESIS OF PERCEPTION & FORM
        </span>
      </div>
    </div>
  );
}

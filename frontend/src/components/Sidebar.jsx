import React from 'react';
import { BookOpen, Scan, Clock, BarChart3, Info } from 'lucide-react';

export default function Sidebar({ activeTab, onTabChange }) {
  const navItems = [
    { id: 'landing', label: 'Story', icon: BookOpen },
    { id: 'detect', label: 'Detect', icon: Scan },
    { id: 'history', label: 'History', icon: Clock },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'about', label: 'About', icon: Info },
  ];

  return (
    <aside style={styles.sidebar}>
      {/* Top Brand Wireframe Icon (Click returns to Landing) */}
      <div
        style={styles.brandContainer}
        onClick={() => onTabChange('landing')}
        title="Engine Intelligence — Return to Editorial Story"
        className="interactive-target"
      >
        <div style={styles.logoFrame}>
          <svg
            width="54"
            height="20"
            viewBox="0 0 52 18"
            fill="none"
            stroke="var(--accent-petrol-teal)"
            strokeWidth="1.4"
            style={{ display: 'block', margin: '0 auto' }}
          >
            <path d="M4 14C8 14 10 11 14 11H38C42 11 44 14 48 14M10 11L14 4H36L42 11M8 14A3 3 0 1 1 8 13.99M44 14A3 3 0 1 1 44 13.99" />
          </svg>
          <span style={styles.brandMicroText}>ENGINE INTEL</span>
        </div>
      </div>

      {/* Nav Menu Items */}
      <nav style={styles.nav}>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              style={{
                ...styles.navButton,
                ...(isActive ? styles.navButtonActive : styles.navButtonInactive),
              }}
              title={item.label}
              className="interactive-target"
            >
              <Icon size={18} strokeWidth={isActive ? 2.2 : 1.6} />
              <span style={styles.navLabel}>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Bottom Editorial Tag */}
      <div style={styles.footer}>
        <span style={styles.footerText}>CV Hackathon</span>
        <div style={styles.footerSub}>
          <span>'26</span>
          <span style={styles.footerSlash}>/</span>
        </div>
      </div>
    </aside>
  );
}

const styles = {
  sidebar: {
    width: '136px',
    backgroundColor: 'var(--bg-charcoal)',
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    padding: '24px 14px',
    borderRight: '1px solid var(--border-dark)',
    flexShrink: 0,
    zIndex: 20,
  },
  brandContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    paddingBottom: '20px',
    borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
    cursor: 'pointer',
  },
  logoFrame: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '6px',
  },
  brandMicroText: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.58rem',
    letterSpacing: '0.14em',
    color: 'var(--text-light-muted)',
    fontWeight: '700',
  },
  nav: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    margin: 'auto 0',
  },
  navButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    width: '100%',
    padding: '10px 12px',
    borderRadius: '12px',
    border: 'none',
    cursor: 'pointer',
    transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
    fontFamily: 'var(--font-sans)',
    fontSize: '0.84rem',
    textAlign: 'left',
  },
  navButtonActive: {
    backgroundColor: 'var(--accent-petrol-teal)',
    color: '#FFFFFF',
    fontWeight: '600',
    boxShadow: '0 4px 14px rgba(23, 107, 112, 0.38)',
  },
  navButtonInactive: {
    backgroundColor: 'transparent',
    color: 'var(--text-light-muted)',
    fontWeight: '400',
  },
  navLabel: {
    letterSpacing: '0.02em',
  },
  footer: {
    paddingTop: '20px',
    borderTop: '1px solid rgba(255, 255, 255, 0.08)',
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  footerText: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.66rem',
    color: 'var(--text-light-muted)',
    letterSpacing: '0.04em',
    opacity: 0.75,
  },
  footerSub: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.74rem',
    color: 'var(--text-light-bright)',
    fontWeight: '700',
  },
  footerSlash: {
    color: 'var(--accent-tangerine)',
    fontSize: '0.88rem',
    fontWeight: '700',
  },
};

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
    <>
      {/* Desktop / Tablet Left Charcoal Sidebar */}
      <aside className="sidebar-desktop">
        {/* Top Brand Wireframe Icon (Click returns to Landing) */}
        <div
          className="sidebar-brand"
          onClick={() => onTabChange('landing')}
          title="Engine Intelligence — Return to Editorial Story"
        >
          <div className="sidebar-logo-frame">
            <svg
              width="48"
              height="18"
              viewBox="0 0 52 18"
              fill="none"
              stroke="var(--accent-petrol-teal)"
              strokeWidth="1.6"
              style={{ display: 'block', margin: '0 auto' }}
            >
              <path d="M4 14C8 14 10 11 14 11H38C42 11 44 14 48 14M10 11L14 4H36L42 11M8 14A3 3 0 1 1 8 13.99M44 14A3 3 0 1 1 44 13.99" />
            </svg>
            <span className="sidebar-brand-text">ENGINE INTEL</span>
          </div>
        </div>

        {/* Nav Menu Items */}
        <nav className="sidebar-nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`sidebar-nav-btn ${isActive ? 'active' : ''}`}
                title={item.label}
              >
                <Icon size={18} strokeWidth={isActive ? 2.2 : 1.6} />
                <span className="sidebar-nav-label">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Bottom Editorial Tag */}
        <div className="sidebar-footer">
          <span className="sidebar-footer-text">CV Hackathon</span>
          <div className="sidebar-footer-sub">
            <span>'26</span>
            <span className="sidebar-footer-slash">/</span>
          </div>
        </div>
      </aside>

      {/* Mobile Fixed Bottom Navigation Bar (<= 768px) */}
      <nav className="mobile-bottom-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              className={`mobile-nav-btn ${isActive ? 'active' : ''}`}
              title={item.label}
            >
              <Icon size={18} strokeWidth={isActive ? 2.2 : 1.6} />
              <span className="mobile-nav-label">{item.label}</span>
            </button>
          );
        })}
      </nav>
    </>
  );
}

import React from 'react';
import { X, Clock, BarChart2, BookOpen, ExternalLink, Cpu, CheckCircle } from 'lucide-react';

export default function InfoModals({ activeTab, onClose, historyItems }) {
  if (activeTab === 'detect' || activeTab === 'landing' || !activeTab) return null;

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div style={styles.header}>
          <div style={styles.headerTitle}>
            {activeTab === 'history' && (
              <>
                <Clock size={20} color="var(--accent-petrol-teal)" />
                <span>Detection Archive & History</span>
              </>
            )}
            {activeTab === 'analytics' && (
              <>
                <BarChart2 size={20} color="var(--accent-petrol-teal)" />
                <span>Model Perception Analytics</span>
              </>
            )}
            {activeTab === 'about' && (
              <>
                <BookOpen size={20} color="var(--accent-petrol-teal)" />
                <span>About Engine Intelligence</span>
              </>
            )}
          </div>

          <button style={styles.closeBtn} onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        {/* Tab Content */}
        <div style={styles.body}>
          {activeTab === 'history' && (
            <div style={styles.contentBlock}>
              <p style={styles.introParagraph}>
                Chronological record of processed vehicle imagery, feature extractions, and multi-model consensus reports.
              </p>

              {(!historyItems || historyItems.length === 0) ? (
                <div style={styles.emptyState}>
                  <span style={styles.emptyText}>No historical detections recorded yet in this session.</span>
                </div>
              ) : (
                <div style={styles.historyList}>
                  {historyItems.map((item, i) => (
                    <div key={i} style={styles.historyCard}>
                      <div style={styles.historyHeader}>
                        <span style={styles.historyMake}>{item.make} {item.model}</span>
                        <span style={styles.historyConf}>{(item.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <span style={styles.historyTimestamp}>{item.timestamp}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'analytics' && (
            <div style={styles.contentBlock}>
              <p style={styles.introParagraph}>
                Live diagnostic telemetry of the active neural architecture branches and perception pipelines:
              </p>

              <div style={styles.analyticsGrid}>
                <div style={styles.statCard}>
                  <span style={styles.statNum}>196</span>
                  <span style={styles.statLabel}>Stanford Cars Classes</span>
                </div>
                <div style={styles.statCard}>
                  <span style={styles.statNum}>23</span>
                  <span style={styles.statLabel}>Exterior Seg Components</span>
                </div>
                <div style={styles.statCard}>
                  <span style={styles.statNum}>4</span>
                  <span style={styles.statLabel}>Parallel Vision Branches</span>
                </div>
                <div style={styles.statCard}>
                  <span style={styles.statNum}>INR</span>
                  <span style={styles.statLabel}>Market Valuation Engine</span>
                </div>
              </div>

              <div style={styles.modelStatusBox}>
                <div style={styles.statusRow}>
                  <CheckCircle size={14} color="#176B70" />
                  <span>Branch 1: YOLOv8 Detector + ResNet50 Classifier</span>
                </div>
                <div style={styles.statusRow}>
                  <CheckCircle size={14} color="#176B70" />
                  <span>Branch 2: YOLO11n-seg Carparts Segmentation</span>
                </div>
                <div style={styles.statusRow}>
                  <CheckCircle size={14} color="#176B70" />
                  <span>Branch 3: Groq Vision Primary & Failover Client</span>
                </div>
                <div style={styles.statusRow}>
                  <CheckCircle size={14} color="#176B70" />
                  <span>Final Intelligence: Groq Cross-Branch Verification</span>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'about' && (
            <div style={styles.contentBlock}>
              <h3 style={styles.aboutHeading}>Aesthetic Manifesto: Editorial Surrealism</h3>
              <p style={styles.introParagraph}>
                ENGINE INTELLIGENCE re-envisions automotive artificial intelligence as an intersection between
                experimental contemporary art, vintage technical publication, and high-performance mechanical engineering.
              </p>
              <p style={styles.bodyParagraph}>
                Constructed for the CV Hackathon '26, the architecture incorporates classical marble sculpture iconography,
                duotone engine visors, and floating alpine viaducts, rejecting generic SaaS AI tropes in favor of tactile editorial beauty.
              </p>

              <div style={styles.attributionBox}>
                <span style={styles.attrTitle}>TECHNICAL STACK</span>
                <span style={styles.attrDetail}>FastAPI • PyTorch 2.10 • YOLO11n-seg • Groq Llama 3.2 Vision • React 19 • Vite</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const styles = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100vw',
    height: '100vh',
    backgroundColor: 'rgba(21, 21, 19, 0.65)',
    backdropFilter: 'blur(4px)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 100,
    padding: '20px',
  },
  modal: {
    width: '100%',
    maxWidth: '540px',
    backgroundColor: '#F1EBDD',
    borderRadius: '24px',
    border: '1px solid rgba(32, 32, 30, 0.15)',
    boxShadow: '0 24px 60px rgba(0, 0, 0, 0.25)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    animation: 'fadeIn 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '20px 24px',
    borderBottom: '1px solid rgba(32, 32, 30, 0.1)',
    backgroundColor: '#EBE4D5',
  },
  headerTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontFamily: 'var(--font-sans)',
    fontSize: '1rem',
    fontWeight: '700',
    color: 'var(--text-ink-black)',
  },
  closeBtn: {
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    color: 'var(--text-ink-muted)',
    padding: '4px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  body: {
    padding: '24px',
  },
  contentBlock: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  introParagraph: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.88rem',
    color: 'var(--text-ink-muted)',
    lineHeight: 1.5,
  },
  bodyParagraph: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.84rem',
    color: 'var(--text-ink-muted)',
    lineHeight: 1.5,
  },
  aboutHeading: {
    fontFamily: 'var(--font-editorial)',
    fontSize: '1.4rem',
    fontWeight: '700',
    color: 'var(--text-ink-black)',
  },
  analyticsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '12px',
    margin: '8px 0',
  },
  statCard: {
    backgroundColor: '#EBE4D5',
    padding: '16px',
    borderRadius: '16px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '4px',
    border: '1px solid rgba(32, 32, 30, 0.08)',
  },
  statNum: {
    fontFamily: 'var(--font-editorial)',
    fontSize: '2rem',
    fontWeight: '700',
    color: 'var(--accent-petrol-teal)',
  },
  statLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.68rem',
    color: 'var(--text-ink-muted)',
    textAlign: 'center',
  },
  modelStatusBox: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    backgroundColor: 'rgba(255, 255, 255, 0.45)',
    padding: '12px 16px',
    borderRadius: '12px',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    color: 'var(--text-ink-black)',
  },
  statusRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  attributionBox: {
    marginTop: '12px',
    padding: '12px',
    backgroundColor: '#EBE4D5',
    borderRadius: '12px',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  attrTitle: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.66rem',
    color: 'var(--text-ink-muted)',
    fontWeight: '700',
    letterSpacing: '0.08em',
  },
  attrDetail: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    color: 'var(--accent-petrol-teal)',
    fontWeight: '600',
  },
  emptyState: {
    padding: '32px 16px',
    textAlign: 'center',
    backgroundColor: '#EBE4D5',
    borderRadius: '16px',
  },
  emptyText: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.78rem',
    color: 'var(--text-ink-faint)',
  },
  historyList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  historyCard: {
    padding: '12px 14px',
    backgroundColor: '#EBE4D5',
    borderRadius: '12px',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  historyHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  historyMake: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.9rem',
    fontWeight: '700',
    color: 'var(--text-ink-black)',
  },
  historyConf: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.8rem',
    fontWeight: '700',
    color: 'var(--accent-terracotta)',
  },
  historyTimestamp: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.68rem',
    color: 'var(--text-ink-faint)',
  },
};

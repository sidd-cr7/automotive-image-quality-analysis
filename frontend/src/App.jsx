import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import LandingView from './components/LandingView';
import DetectView from './components/DetectView';
import InfoModals from './components/InfoModals';
import CustomCursor from './components/CustomCursor';
import ScrollProgress from './components/ScrollProgress';
import Preloader from './components/Preloader';

export default function App() {
  const [activeTab, setActiveTab] = useState('landing');
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [resultData, setResultData] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [historyItems, setHistoryItems] = useState([]);
  const [isPreloaded, setIsPreloaded] = useState(false);

  // Handle image selection in workspace
  const handleImageSelected = (file) => {
    setSelectedFile(file);
    setErrorMessage(null);
    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);
  };

  // Reset upload state
  const handleReset = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setErrorMessage(null);
  };

  // Call the live AUTOVISION AI FastAPI backend
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    // Auto-detect API base URL (relative for proxy / port 8000 fallback)
    const apiBase = window.location.port === '5173' ? 'http://localhost:8000' : '';
    const endpoint = `${apiBase}/predict?market_region=India&enable_groq_inspection=true&enable_final_verdict=true`;

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned error status ${response.status}`);
      }

      const data = await response.json();
      setResultData(data);

      // Add to session history
      const newHistoryItem = {
        make: data?.vehicle?.make || 'Vehicle',
        model: data?.vehicle?.model || '',
        confidence: data?.vehicle?.confidence || 0.0,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setHistoryItems((prev) => [newHistoryItem, ...prev]);

    } catch (err) {
      console.warn('Backend inference failed:', err);
      setErrorMessage(`Live inference note: ${err.message}. Showing real system response status.`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Modal close handler: returns to previous main tab ('landing' or 'detect')
  const handleModalClose = () => {
    if (activeTab === 'history' || activeTab === 'analytics' || activeTab === 'about') {
      setActiveTab('detect');
    }
  };

  return (
    <div style={styles.appRoot}>
      {/* 1. Cinematic Introductory Preloader (First Visit Only) */}
      <Preloader onComplete={() => setIsPreloaded(true)} />

      {/* 2. Top Scroll Progress Indicator */}
      <ScrollProgress />

      {/* 3. Circular Custom Cursor with Trailing Ring */}
      <CustomCursor />

      {/* 4. Left Charcoal Sidebar */}
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />

      {/* 5. Main Canvas (Landing View or Detection View) */}
      <main style={styles.mainCanvas}>
        {activeTab === 'landing' ? (
          <LandingView onNavigateToDetect={() => setActiveTab('detect')} />
        ) : (
          <DetectView
            onNavigateToLanding={() => setActiveTab('landing')}
            onImageSelected={handleImageSelected}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
            selectedFile={selectedFile}
            imagePreview={imagePreview}
            onReset={handleReset}
            errorMessage={errorMessage}
            resultData={resultData}
          />
        )}
      </main>

      {/* 6. Informational Overlays for History, Analytics, About */}
      <InfoModals
        activeTab={activeTab}
        onClose={handleModalClose}
        historyItems={historyItems}
      />
    </div>
  );
}

const styles = {
  appRoot: {
    display: 'flex',
    minHeight: '100vh',
    width: '100%',
    backgroundColor: 'var(--bg-warm-chalk)',
    color: 'var(--text-ink-black)',
    position: 'relative',
    overflowX: 'hidden',
  },
  mainCanvas: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    padding: '36px 48px',
    maxWidth: '1680px',
    margin: '0 auto',
    width: '100%',
    position: 'relative',
    zIndex: 10,
  },
};

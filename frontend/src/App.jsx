import React, { useEffect, useState } from 'react';
import Sidebar from './components/Sidebar';
import LandingView from './components/LandingView';
import DetectView from './components/DetectView';
import InfoModals from './components/InfoModals';
import ScrollProgress from './components/ScrollProgress';

// Elements that fade/slide in on scroll, and headings that get the text-reveal effect.
const REVEAL_SELECTOR = [
  '.hero-artwork-frame', '.hero-editorial-footer', '.ei-head', '.ei-branch', '.ei-q > *',
  '.ei-spec-layout > *', '.ei-cta', '.detect-header-block', '.detect-upload-column', '.detect-results-column',
].join(',');
const TEXT_SELECTOR = '.hero-title-line, .ei-h2, .ei-h3, .ei-cta-title, .detect-page-title';

export default function App() {
  const getInitialTab = () => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      if (params.get('tab') === 'detect' || window.location.hash === '#detect') {
        return 'detect';
      }
    }
    return 'landing';
  };

  const [activeTab, setActiveTab] = useState(getInitialTab);

  // Scroll-reveal: CSS hides `.reveal` elements only when motion is allowed (see landing.css).
  useEffect(() => {
    window.scrollTo(0, 0);
    if (typeof IntersectionObserver === 'undefined') return undefined;
    const nodes = document.querySelectorAll(`${REVEAL_SELECTOR}, ${TEXT_SELECTOR}`);
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add('is-visible');
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.06, rootMargin: '0px 0px -6% 0px' },
    );
    nodes.forEach((n) => {
      n.classList.add(n.matches(TEXT_SELECTOR) ? 'reveal-text' : 'reveal');
      io.observe(n);
    });
    // Safety net for jumps (End key, anchors): reveal anything now at or above the fold.
    let timer = 0;
    const sweep = () => {
      document.querySelectorAll('.reveal:not(.is-visible), .reveal-text:not(.is-visible)').forEach((n) => {
        if (n.getBoundingClientRect().top < window.innerHeight * 0.94) {
          n.classList.add('is-visible');
          io.unobserve(n);
        }
      });
    };
    const onScroll = () => {
      clearTimeout(timer);
      timer = setTimeout(sweep, 140);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      clearTimeout(timer);
      window.removeEventListener('scroll', onScroll);
      io.disconnect();
    };
  }, [activeTab]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [resultData, setResultData] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [historyItems, setHistoryItems] = useState([]);
  const [analysis, setAnalysis] = useState(null);

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
    setResultData(null);
    setErrorMessage(null);
  };

  // Call the live FastAPI backend. Uses XHR for real upload progress and polls
  // /progress/{jobId} for real server-side stage boundaries (nothing is simulated).
  const handleAnalyze = () => {
    if (!selectedFile || isAnalyzing) return;

    const jobId = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
    const apiBase = window.location.port === '5173' ? 'http://localhost:8000' : '';
    const endpoint = `${apiBase}/predict?market_region=India&enable_groq_inspection=true&enable_final_verdict=true&job_id=${jobId}`;

    setIsAnalyzing(true);
    setErrorMessage(null);
    setResultData(null);
    setAnalysis({ startedAt: Date.now(), uploadPct: 0, uploadDone: false, current: null, completed: [] });

    const formData = new FormData();
    formData.append('file', selectedFile);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', endpoint);
    xhr.timeout = 300000;

    const poll = setInterval(async () => {
      try {
        const r = await fetch(`${apiBase}/progress/${jobId}`);
        if (r.ok) {
          const p = await r.json();
          setAnalysis((a) => (a ? { ...a, current: p.current, completed: p.completed } : a));
        }
      } catch {
        /* progress is best-effort; the main request reports real errors */
      }
    }, 1000);

    const end = (message) => {
      clearInterval(poll);
      setIsAnalyzing(false);
      setAnalysis(null);
      if (message) setErrorMessage(message);
    };

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        setAnalysis((a) => (a ? { ...a, uploadPct: Math.round((e.loaded / e.total) * 100) } : a));
      }
    };
    xhr.upload.onload = () => setAnalysis((a) => (a ? { ...a, uploadPct: 100, uploadDone: true } : a));

    xhr.onload = () => {
      let data = null;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        /* handled below */
      }
      if (xhr.status < 200 || xhr.status >= 300 || !data) {
        const detail = data?.detail;
        const text = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `Server returned status ${xhr.status}`;
        end(`Analysis failed: ${text}`);
        return;
      }
      setResultData(data);
      setHistoryItems((prev) => [
        {
          make: data?.vehicle?.make || 'Vehicle',
          model: data?.vehicle?.model || '',
          confidence: data?.vehicle?.confidence ?? null,
          uncertain: data?.vehicle?.is_uncertain === true,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
        ...prev,
      ]);
      end();
    };
    xhr.onerror = () => end('Analysis failed: could not reach the analysis server. Check that the backend is running.');
    xhr.ontimeout = () => end('Analysis failed: the request timed out after 5 minutes. The vision service may be overloaded; try again shortly.');
    xhr.onabort = () => end('Analysis cancelled.');

    xhr.send(formData);
  };

  // Modal close handler: returns to previous main tab ('landing' or 'detect')
  const handleModalClose = () => {
    if (activeTab === 'history' || activeTab === 'analytics' || activeTab === 'about') {
      setActiveTab('detect');
    }
  };

  return (
    <div className="app-layout">
      <ScrollProgress />
      {/* 1. Left Charcoal Navigation Sidebar (Responsive: Bottom bar on mobile) */}
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />

      {/* 2. Main Canvas (Landing View or Detection View) */}
      <main className="app-main-content">
        <div key={activeTab === 'landing' ? 'landing' : 'detect'} className="page-enter">
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
            analysis={analysis}
          />
        )}
        </div>
      </main>

      {/* 3. Informational Overlays for History, Analytics, About */}
      <InfoModals
        activeTab={activeTab}
        onClose={handleModalClose}
        historyItems={historyItems}
      />
    </div>
  );
}

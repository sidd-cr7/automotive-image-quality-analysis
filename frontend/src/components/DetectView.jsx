import React from 'react';
import { ArrowLeft, Sparkles, Terminal } from 'lucide-react';
import UploadArea from './UploadArea';
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
  analysis,
}) {
  return (
    <div className="detect-container">
      {/* Top Workspace Navigation Bar */}
      <header className="detect-top-bar">
        <button
          onClick={onNavigateToLanding}
          className="btn-editorial-outline btn-back"
          title="Return to Landing Page"
        >
          <ArrowLeft size={16} />
          <span>RETURN TO EDITORIAL STORY</span>
        </button>

        <div className="workspace-status-badge">
          <span className="status-live-dot" />
          <span className="status-live-text">3 BRANCHES ONLINE</span>
        </div>
      </header>

      {/* Workspace Editorial Header */}
      <div className="detect-header-block">
        <div className="detect-meta-tag">WORKSPACE 01 — MULTI-MODEL VEHICLE PERCEPTION</div>
        <h1 className="detect-page-title">Visual Inference Laboratory</h1>
        <p className="detect-page-subtext">
          Submit vehicle exteriors, engine bays, or cabin cockpit photographs. The platform
          executes mathematical defect quantification, adaptive lighting optimization, and
          multi-branch neural inference simultaneously.
        </p>
      </div>

      {/* 2-Column Responsive Workspace (Upload + Real Results) */}
      <div className="detect-workspace-layout">
        {/* Left Column: Exactly ONE Upload Area */}
        <section className="detect-upload-column">
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

        {/* Right Column: Genuine Results Area (Empty standby when idle, genuine data after inference) */}
        <section className="detect-results-column">
          <ResultsArea resultData={resultData} isLoading={isAnalyzing} analysis={analysis} />
        </section>
      </div>
    </div>
  );
}

import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, CheckCircle, RefreshCw, AlertCircle, Sparkles, X } from 'lucide-react';

export default function UploadArea({
  onImageSelected,
  onAnalyze,
  isAnalyzing,
  selectedFile,
  imagePreview,
  onReset,
  errorMessage,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPEG, PNG, WEBP, BMP).');
      return;
    }
    onImageSelected(file);
  };

  const handleSampleClick = () => {
    // Generate a clean test canvas vehicle image for instant testing
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 300;
    const ctx = canvas.getContext('2d');
    
    // Warm vintage gradient background
    const grad = ctx.createLinearGradient(0, 0, 400, 300);
    grad.addColorStop(0, '#EAE3D2');
    grad.addColorStop(1, '#C8BDA5');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 400, 300);

    // Draw stylized mechanical vehicle outline
    ctx.fillStyle = '#176B70';
    ctx.fillRect(70, 110, 260, 80);
    ctx.fillStyle = '#20201E';
    ctx.beginPath();
    ctx.arc(120, 190, 26, 0, Math.PI * 2);
    ctx.arc(280, 190, 26, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#F28B45';
    ctx.font = 'bold 15px "Space Mono", monospace';
    ctx.fillText('AUTOMOTIVE TEST PATTERN', 80, 155);

    canvas.toBlob((blob) => {
      const sampleFile = new File([blob], 'automotive_test_specimen.jpg', { type: 'image/jpeg' });
      onImageSelected(sampleFile);
    }, 'image/jpeg', 0.95);
  };

  return (
    <div style={styles.wrapper}>
      {/* Exactly ONE Upload Dropzone Container */}
      <div
        className={`upload-dropzone interactive-target ${isDragging ? 'active-drag' : ''}`}
        style={styles.card}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !imagePreview && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          style={{ display: 'none' }}
          onChange={handleFileInput}
        />

        {/* Top Right Landscape Corner Icon */}
        <div style={styles.topRightIcon}>
          <ImageIcon size={22} color="var(--text-ink-black)" strokeWidth={1.4} opacity={0.65} />
        </div>

        {/* Empty State / Upload Prompt */}
        {!imagePreview ? (
          <div style={styles.promptContent}>
            <div style={styles.iconCircle}>
              <UploadCloud size={30} color="var(--accent-petrol-teal)" strokeWidth={1.8} />
            </div>

            <h3 style={styles.title}>Upload Vehicle or Engine</h3>
            <p style={styles.subtitle}>Drag & drop or click to browse</p>

            <div style={styles.formatBadge}>
              <span>JPEG, PNG, WEBP, BMP up to 20MB</span>
            </div>
          </div>
        ) : (
          /* Image Selected / Preview State */
          <div style={styles.previewContent} onClick={(e) => e.stopPropagation()}>
            <div style={styles.imageThumbnailFrame}>
              <img src={imagePreview} alt="Selected preview" style={styles.thumbnailImg} />
              <div style={styles.selectedBadge}>
                <CheckCircle size={13} color="#FFFFFF" />
                <span>Ready for Inference</span>
              </div>
            </div>

            <div style={styles.fileInfo}>
              <span style={styles.fileName}>{selectedFile?.name || 'automotive_specimen.jpg'}</span>
              <span style={styles.fileSize}>
                {selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : 'Active input'}
              </span>
            </div>

            {/* Action Buttons */}
            <div style={styles.actionRow}>
              <button
                className="btn-editorial-teal interactive-target"
                onClick={onAnalyze}
                disabled={isAnalyzing}
                style={{
                  ...styles.analyzeBtn,
                  opacity: isAnalyzing ? 0.75 : 1,
                }}
              >
                {isAnalyzing ? (
                  <>
                    <RefreshCw size={15} className="animate-pulse-subtle" />
                    <span>Analyzing Vision...</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={15} />
                    <span>Initiate Analysis</span>
                  </>
                )}
              </button>

              <button
                style={styles.changeBtn}
                onClick={onReset}
                disabled={isAnalyzing}
                title="Change Image"
                className="interactive-target"
              >
                <X size={14} />
                <span>Reset</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Quick Test Sample Helper */}
      {!imagePreview && (
        <div style={styles.quickSampleRow}>
          <span style={styles.sampleHint}>No image ready?</span>
          <button style={styles.sampleBtn} onClick={handleSampleClick} className="interactive-target">
            Load Test Specimen
          </button>
        </div>
      )}

      {/* Error Banner */}
      {errorMessage && (
        <div style={styles.errorBanner}>
          <AlertCircle size={16} color="var(--accent-tangerine)" />
          <span style={styles.errorText}>{errorMessage}</span>
        </div>
      )}
    </div>
  );
}

const styles = {
  wrapper: {
    width: '100%',
    maxWidth: '380px',
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    zIndex: 15,
  },
  card: {
    position: 'relative',
    width: '100%',
    minHeight: '270px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '32px 24px',
    backgroundColor: 'rgba(235, 228, 213, 0.55)',
    boxShadow: '0 8px 30px rgba(32, 32, 30, 0.04)',
  },
  topRightIcon: {
    position: 'absolute',
    top: '18px',
    right: '20px',
  },
  promptContent: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
    gap: '8px',
  },
  iconCircle: {
    width: '56px',
    height: '56px',
    borderRadius: '50%',
    backgroundColor: 'rgba(23, 107, 112, 0.08)',
    border: '1px solid var(--accent-petrol-teal-border)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: '8px',
  },
  title: {
    fontFamily: 'var(--font-editorial)',
    fontSize: '1.5rem',
    fontWeight: '700',
    color: 'var(--text-ink-black)',
    margin: 0,
  },
  subtitle: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.85rem',
    color: 'var(--text-ink-muted)',
    margin: 0,
  },
  formatBadge: {
    marginTop: '12px',
    padding: '4px 10px',
    borderRadius: 'var(--radius-full)',
    backgroundColor: 'rgba(32, 32, 30, 0.05)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.64rem',
    letterSpacing: '0.06em',
    color: 'var(--text-ink-muted)',
  },
  previewContent: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '12px',
  },
  imageThumbnailFrame: {
    position: 'relative',
    width: '100%',
    maxHeight: '180px',
    borderRadius: 'var(--radius-md)',
    overflow: 'hidden',
    border: '1px solid var(--border-subtle)',
    backgroundColor: '#000',
  },
  thumbnailImg: {
    width: '100%',
    height: '180px',
    objectFit: 'cover',
    display: 'block',
  },
  selectedBadge: {
    position: 'absolute',
    bottom: '8px',
    left: '8px',
    backgroundColor: 'rgba(23, 107, 112, 0.92)',
    backdropFilter: 'blur(4px)',
    color: '#FFFFFF',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.65rem',
    padding: '3px 8px',
    borderRadius: 'var(--radius-full)',
    display: 'flex',
    alignItems: 'center',
    gap: '5px',
    fontWeight: '600',
  },
  fileInfo: {
    width: '100%',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    fontSize: '0.75rem',
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-ink-muted)',
    padding: '0 4px',
  },
  fileName: {
    maxWidth: '180px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  fileSize: {
    opacity: 0.8,
  },
  actionRow: {
    width: '100%',
    display: 'flex',
    gap: '8px',
    marginTop: '4px',
  },
  analyzeBtn: {
    flex: 1,
    padding: '10px 16px',
    fontSize: '0.86rem',
  },
  changeBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    background: 'none',
    border: '1px solid var(--border-medium)',
    borderRadius: 'var(--radius-full)',
    padding: '8px 14px',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    color: 'var(--text-ink-muted)',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  },
  quickSampleRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    fontSize: '0.74rem',
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-ink-muted)',
  },
  sampleHint: {
    opacity: 0.8,
  },
  sampleBtn: {
    background: 'none',
    border: 'none',
    color: 'var(--accent-petrol-teal)',
    textDecoration: 'underline',
    cursor: 'pointer',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.74rem',
    fontWeight: '600',
  },
  errorBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 14px',
    backgroundColor: 'rgba(242, 139, 69, 0.1)',
    border: '1px solid var(--accent-tangerine-border)',
    borderRadius: 'var(--radius-md)',
    fontFamily: 'var(--font-sans)',
    fontSize: '0.78rem',
    color: 'var(--text-ink-black)',
  },
  errorText: {
    flex: 1,
    lineHeight: 1.4,
  },
};

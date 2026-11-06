import React, { useEffect, useState, useRef } from 'react';

export default function CustomCursor() {
  const [position, setPosition] = useState({ x: -100, y: -100 });
  const [followerPos, setFollowerPos] = useState({ x: -100, y: -100 });
  const [isHovering, setIsHovering] = useState(false);
  const [isClicking, setIsClicking] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [isTouchDevice, setIsTouchDevice] = useState(false);

  const posRef = useRef({ x: -100, y: -100 });
  const followerRef = useRef({ x: -100, y: -100 });
  const rafId = useRef(null);

  useEffect(() => {
    // Check if device is touch-primary
    const isTouch = window.matchMedia('(pointer: coarse)').matches || 'ontouchstart' in window;
    setIsTouchDevice(isTouch);
    if (isTouch) return;

    document.body.classList.add('has-custom-cursor');

    const handleMouseMove = (e) => {
      posRef.current = { x: e.clientX, y: e.clientY };
      setPosition({ x: e.clientX, y: e.clientY });
      setIsVisible(true);
    };

    const handleMouseDown = () => setIsClicking(true);
    const handleMouseUp = () => setIsClicking(false);
    const handleMouseLeave = () => setIsVisible(false);
    const handleMouseEnter = () => setIsVisible(true);

    // Track hover over interactive targets
    const handleElementHover = (e) => {
      const target = e.target;
      const isInteractive = target.closest('button, a, input, select, .upload-dropzone, [role="button"], .interactive-target');
      setIsHovering(!!isInteractive);
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    window.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('mouseleave', handleMouseLeave);
    document.addEventListener('mouseenter', handleMouseEnter);
    window.addEventListener('mouseover', handleElementHover, { passive: true });

    // Smooth Lerp loop for trailing outer ring
    const smoothFollower = () => {
      const dx = posRef.current.x - followerRef.current.x;
      const dy = posRef.current.y - followerRef.current.y;
      followerRef.current.x += dx * 0.18;
      followerRef.current.y += dy * 0.18;

      setFollowerPos({
        x: followerRef.current.x,
        y: followerRef.current.y,
      });

      rafId.current = requestAnimationFrame(smoothFollower);
    };
    rafId.current = requestAnimationFrame(smoothFollower);

    return () => {
      document.body.classList.remove('has-custom-cursor');
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('mouseleave', handleMouseLeave);
      document.removeEventListener('mouseenter', handleMouseEnter);
      window.removeEventListener('mouseover', handleElementHover);
      if (rafId.current) cancelAnimationFrame(rafId.current);
    };
  }, []);

  if (isTouchDevice || !isVisible) return null;

  return (
    <>
      {/* 1. Precise Center Dot */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          pointerEvents: 'none',
          zIndex: 100000,
          transform: `translate3d(${position.x}px, ${position.y}px, 0) translate(-50%, -50%) scale(${isClicking ? 0.7 : isHovering ? 1.4 : 1})`,
          transition: 'transform 0.12s ease-out, background-color 0.2s ease',
          width: '7px',
          height: '7px',
          borderRadius: '50%',
          backgroundColor: isHovering ? 'var(--accent-tangerine)' : 'var(--accent-petrol-teal)',
          boxShadow: isHovering ? '0 0 10px rgba(242, 139, 69, 0.6)' : '0 0 8px rgba(23, 107, 112, 0.4)',
        }}
      />

      {/* 2. Trailing Smooth Outer Ring */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          pointerEvents: 'none',
          zIndex: 99999,
          transform: `translate3d(${followerPos.x}px, ${followerPos.y}px, 0) translate(-50%, -50%) scale(${isHovering ? 1.6 : isClicking ? 0.85 : 1})`,
          transition: 'transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.2s ease, background-color 0.2s ease',
          width: '34px',
          height: '34px',
          borderRadius: '50%',
          border: `1.5px solid ${isHovering ? 'var(--accent-tangerine)' : 'var(--accent-petrol-teal)'}`,
          backgroundColor: isHovering ? 'rgba(242, 139, 69, 0.08)' : 'transparent',
          opacity: 0.85,
        }}
      />
    </>
  );
}

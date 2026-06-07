import { useRef, useEffect, useCallback, useState } from 'react';
import { renderFieldToImageData, COLORMAP_NAMES, RENDER_W, RENDER_H } from '../utils/colormap';
import { sampleField, physToGrid, canvasToPhys, X_MIN, X_MAX, Y_MIN, Y_MAX } from '../utils/decode';

const GRID = 256;
const PARTICLE_COUNT = 1800;
const MAX_TRAIL = 18;
const PARTICLE_SPEED = 0.4;
const MAX_AGE = 120;

/** Create a particle at a random position in the domain */
function spawnParticle() {
  return {
    x: X_MIN + Math.random() * (X_MAX - X_MIN),
    y: Y_MIN + Math.random() * (Y_MAX - Y_MIN),
    trail: [],
    age: Math.floor(Math.random() * MAX_AGE),
  };
}

export default function FieldRenderer({
  fieldData,       // Float32Array (256*256)
  velocityX,       // Float32Array (256*256) — for particles
  velocityY,       // Float32Array (256*256)
  sdf,             // Float32Array (256*256)
  vmin,
  vmax,
  colormap = 'coolwarm',
  label = 'Field',
  showParticles = true,
  animating = true,
}) {
  const heatmapCanvasRef = useRef(null);
  const particleCanvasRef = useRef(null);
  const containerRef = useRef(null);
  const particlesRef = useRef(null);
  const animFrameRef = useRef(null);
  const prevFieldRef = useRef(null);
  const transitionRef = useRef({ progress: 1, oldField: null, newField: null });
  const blendBufRef = useRef(null);  // reused transition blend buffer (avoids per-frame alloc)
  const [hover, setHover] = useState(null);
  const [displaySize, setDisplaySize] = useState({ w: 768, h: 512 });

  // Observe container size
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(entries => {
      const { width } = entries[0].contentRect;
      // Maintain 3:2 aspect ratio (domain is x:[-1,2]=3, y:[-1,1]=2)
      const w = Math.floor(width);
      const h = Math.floor(w * 2 / 3);
      setDisplaySize({ w, h });
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Initialize particles
  useEffect(() => {
    if (!particlesRef.current) {
      particlesRef.current = Array.from({ length: PARTICLE_COUNT }, spawnParticle);
    }
  }, []);

  // Trigger smooth transition when field data changes
  useEffect(() => {
    if (!fieldData) return;
    if (prevFieldRef.current && prevFieldRef.current !== fieldData) {
      transitionRef.current = {
        progress: 0,
        oldField: prevFieldRef.current,
        newField: fieldData,
      };
    } else {
      transitionRef.current = { progress: 1, oldField: null, newField: fieldData };
    }
    prevFieldRef.current = fieldData;
  }, [fieldData]);

  // Render heatmap (static or transitioning)
  const renderHeatmap = useCallback((currentField) => {
    const canvas = heatmapCanvasRef.current;
    if (!canvas || !currentField) return;
    const ctx = canvas.getContext('2d');
    const imageData = renderFieldToImageData(currentField, vmin, vmax, colormap, sdf);
    canvas.width = RENDER_W;
    canvas.height = RENDER_H;
    ctx.putImageData(imageData, 0, 0);
  }, [vmin, vmax, colormap, sdf]);

  // Main animation loop
  useEffect(() => {
    if (!fieldData || !velocityX || !velocityY) return;

    const particles = particlesRef.current;
    if (!particles) return;

    if (!blendBufRef.current) blendBufRef.current = new Float32Array(GRID * GRID);

    // Redraw the (expensive, ~1.5M px) heatmap only when it actually changes:
    // every frame *during* a transition, then exactly once when it settles.
    // This local resets to false on each effect re-run (field/colormap/vmin/vmax
    // /sdf/size change), so those changes always trigger a fresh draw.
    let heatmapDrawn = false;

    const loop = () => {
      const trans = transitionRef.current;

      if (trans.progress < 1 && trans.oldField && trans.newField) {
        trans.progress = Math.min(1, trans.progress + 0.04); // ~25 frames = ~400ms
        const t = trans.progress;
        const oldF = trans.oldField, newF = trans.newField;
        const blended = blendBufRef.current;
        for (let k = 0; k < blended.length; k++) {
          blended[k] = oldF[k] * (1 - t) + newF[k] * t;
        }
        renderHeatmap(blended);
        heatmapDrawn = false; // force one crisp final draw when the transition ends
      } else if (!heatmapDrawn) {
        renderHeatmap(trans.newField || fieldData);
        trans.oldField = null;
        heatmapDrawn = true;
      }

      // Particle animation
      if (showParticles && animating) {
        const pCanvas = particleCanvasRef.current;
        if (!pCanvas) { animFrameRef.current = requestAnimationFrame(loop); return; }

        const pCtx = pCanvas.getContext('2d');
        const cw = displaySize.w;
        const ch = displaySize.h;

        if (pCanvas.width !== cw || pCanvas.height !== ch) {
          pCanvas.width = cw;
          pCanvas.height = ch;
        }

        // Clear
        pCtx.clearRect(0, 0, cw, ch);

        for (let p = 0; p < particles.length; p++) {
          const particle = particles[p];

          // Get velocity at particle position (grid coords)
          const [gi, gj] = physToGrid(particle.x, particle.y);
          const vx = sampleField(velocityX, gi, gj);
          const vy = sampleField(velocityY, gi, gj);

          // Store trail point
          particle.trail.push({ x: particle.x, y: particle.y });
          if (particle.trail.length > MAX_TRAIL) particle.trail.shift();

          // Move particle (physical coordinates)
          const speed = Math.sqrt(vx * vx + vy * vy);
          const dt = PARTICLE_SPEED / (speed + 0.01);
          particle.x += vx * dt * 0.015;
          particle.y += vy * dt * 0.015;
          particle.age++;

          // Check SDF — reset if inside airfoil
          const [si, sj] = physToGrid(particle.x, particle.y);
          const sdfVal = sdf ? sampleField(sdf, si, sj) : 1;

          // Reset if out of bounds, too old, or inside airfoil
          if (
            particle.x < X_MIN || particle.x > X_MAX ||
            particle.y < Y_MIN || particle.y > Y_MAX ||
            particle.age > MAX_AGE ||
            sdfVal < 0.02
          ) {
            Object.assign(particle, spawnParticle());
            continue;
          }

          // Draw trail
          if (particle.trail.length > 1) {
            pCtx.beginPath();
            const t0 = particle.trail[0];
            const cx0 = ((t0.x - X_MIN) / (X_MAX - X_MIN)) * cw;
            const cy0 = (1 - (t0.y - Y_MIN) / (Y_MAX - Y_MIN)) * ch;
            pCtx.moveTo(cx0, cy0);

            for (let t = 1; t < particle.trail.length; t++) {
              const tp = particle.trail[t];
              const cx = ((tp.x - X_MIN) / (X_MAX - X_MIN)) * cw;
              const cy = (1 - (tp.y - Y_MIN) / (Y_MAX - Y_MIN)) * ch;
              pCtx.lineTo(cx, cy);
            }

            const alpha = Math.min(0.6, speed * 2);
            pCtx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
            pCtx.lineWidth = 0.8;
            pCtx.stroke();
          }
        }
      }

      animFrameRef.current = requestAnimationFrame(loop);
    };

    animFrameRef.current = requestAnimationFrame(loop);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [fieldData, velocityX, velocityY, sdf, showParticles, animating, displaySize, renderHeatmap]);

  // Handle mouse hover
  const handleMouseMove = useCallback((e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const [physX, physY] = canvasToPhys(px, py, rect.width, rect.height);
    const [gi, gj] = physToGrid(physX, physY);
    const i = Math.round(gi), j = Math.round(gj);
    if (i >= 0 && i < 256 && j >= 0 && j < 256) {
      const val = fieldData ? fieldData[i * 256 + j] : 0;
      const sdfVal = sdf ? sdf[i * 256 + j] : 1;
      setHover({
        x: physX.toFixed(3),
        y: physY.toFixed(3),
        value: val.toFixed(5),
        insideAirfoil: sdfVal < 0.015,
        px, py,
      });
    }
  }, [fieldData, sdf]);

  const handleMouseLeave = useCallback(() => setHover(null), []);

  if (!fieldData) {
    return (
      <div className="field-renderer field-renderer--empty" ref={containerRef}>
        <div className="field-renderer__placeholder">
          <span style={{ fontSize: '2rem', opacity: 0.3 }}>🌊</span>
          <span>Waiting for prediction data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="field-renderer" ref={containerRef}>
      <div className="field-renderer__label">{label}</div>
      <div
        className="field-renderer__canvas-wrap"
        style={{ aspectRatio: '3 / 2', position: 'relative' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        {/* Heatmap layer — high-res with bilinear interpolation */}
        <canvas
          ref={heatmapCanvasRef}
          className="field-renderer__heatmap"
          width={RENDER_W}
          height={RENDER_H}
        />
        {/* Particle overlay — at display resolution */}
        <canvas
          ref={particleCanvasRef}
          className="field-renderer__particles"
          width={displaySize.w}
          height={displaySize.h}
        />
        {/* Hover tooltip */}
        {hover && (
          <div
            className="field-renderer__tooltip"
            style={{
              left: Math.min(hover.px + 12, displaySize.w - 160),
              top: hover.py - 50,
            }}
          >
            <div>x: {hover.x}, y: {hover.y}</div>
            <div style={{ fontWeight: 600, color: hover.insideAirfoil ? '#64748b' : '#06b6d4' }}>
              {hover.insideAirfoil ? 'Inside airfoil' : `Value: ${hover.value}`}
            </div>
          </div>
        )}
      </div>
      {/* Colorbar */}
      <div className="field-renderer__colorbar">
        <span className="field-renderer__colorbar-label">{vmin?.toFixed(3)}</span>
        <div className="field-renderer__colorbar-gradient" data-cmap={colormap} />
        <span className="field-renderer__colorbar-label">{vmax?.toFixed(3)}</span>
      </div>
    </div>
  );
}

/**
 * Embeddable HTML + JS for the WebView-based heatmap renderer.
 * Features: colormap LUTs, bilinear upsampling, SDF airfoil rendering,
 * smooth field transitions, particle flow visualization, touch tooltips.
 */

export const RENDERER_HTML = `
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0f1117; overflow: hidden; width: 100vw; height: 100vh; position: relative; }
  #wrap { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; }
  canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }
  #particles { pointer-events: none; }
  #tooltip {
    display: none; position: absolute; padding: 4px 8px; border-radius: 4px;
    background: rgba(15,17,23,0.85); border: 1px solid rgba(255,255,255,0.15);
    color: #94a3b8; font: 10px/1.4 monospace; pointer-events: none; z-index: 10; white-space: nowrap;
  }
  #tooltip.visible { display: block; }
  #tooltip .val { font-weight: 700; color: #06b6d4; }
  #tooltip .airfoil { color: #64748b; }
</style>
</head>
<body>
<div id="wrap">
  <canvas id="c"></canvas>
  <canvas id="particles"></canvas>
  <div id="tooltip"></div>
</div>
<script>
// ============================================================
// Colormap LUTs
// ============================================================
const COLORMAPS = {
  coolwarm: [[59,76,192],[68,90,204],[77,104,215],[87,117,225],[98,130,234],[108,142,241],[119,154,247],[130,165,251],[141,176,254],[152,185,255],[163,194,255],[174,201,253],[184,208,249],[194,213,244],[204,217,238],[213,219,230],[221,221,221],[229,216,209],[236,211,197],[241,204,185],[245,196,173],[247,187,160],[247,177,148],[247,166,135],[244,154,123],[241,141,111],[236,127,99],[229,112,88],[222,96,77],[213,80,66],[203,62,56],[192,40,47],[180,4,38]],
  inferno: [[0,0,4],[2,1,10],[5,3,19],[10,5,30],[17,7,42],[25,9,55],[34,11,67],[44,11,78],[54,11,87],[64,10,95],[75,10,101],[85,10,105],[95,11,108],[105,13,110],[115,15,111],[125,18,110],[135,22,109],[144,27,106],[154,33,103],[163,40,99],[172,48,94],[180,57,89],[187,66,84],[195,76,78],[201,86,73],[207,97,67],[213,108,61],[218,119,55],[222,131,49],[226,143,43],[229,155,37],[232,168,31],[234,181,25],[236,194,20],[236,207,17],[236,220,17],[236,232,26],[237,244,45]],
  viridis: [[68,1,84],[71,13,96],[72,24,106],[72,35,116],[71,46,124],[68,56,130],[64,67,135],[59,77,138],[53,87,140],[47,96,141],[41,105,141],[35,114,141],[30,123,139],[25,132,137],[21,140,135],[18,149,130],[20,157,124],[30,165,117],[46,173,108],[65,180,97],[87,187,86],[111,193,73],[139,199,59],[168,204,46],[197,209,39],[225,213,42],[248,219,72],[253,231,111]],
  plasma: [[13,8,135],[30,6,150],[48,4,162],[65,3,172],[82,2,180],[98,1,185],[114,0,189],[130,0,190],[146,0,189],[161,0,185],[175,4,178],[189,13,169],[200,25,158],[211,38,145],[220,54,131],[228,70,117],[234,87,103],[239,103,90],[243,120,77],[246,137,64],[248,154,52],[249,172,41],[249,189,31],[246,207,24],[242,224,21],[237,241,26],[233,252,67]],
};

const LUT_CACHE = {};
function getLUT(name) {
  if (LUT_CACHE[name]) return LUT_CACHE[name];
  const stops = COLORMAPS[name] || COLORMAPS.coolwarm;
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (stops.length - 1);
    const lo = Math.floor(t), hi = Math.min(lo + 1, stops.length - 1);
    const f = t - lo;
    lut[i*4]   = Math.round(stops[lo][0]*(1-f) + stops[hi][0]*f);
    lut[i*4+1] = Math.round(stops[lo][1]*(1-f) + stops[hi][1]*f);
    lut[i*4+2] = Math.round(stops[lo][2]*(1-f) + stops[hi][2]*f);
    lut[i*4+3] = 255;
  }
  LUT_CACHE[name] = lut;
  return lut;
}

// ============================================================
// Bilinear interpolation + SDF
// ============================================================
const G = 256;
function bilinear(arr, gi, gj) {
  const i0 = gi | 0, j0 = gj | 0;
  const i1 = i0 < 255 ? i0+1 : 255, j1 = j0 < 255 ? j0+1 : 255;
  const fx = gi - i0, fy = gj - j0;
  return arr[i0*G+j0]*(1-fx)*(1-fy) + arr[i1*G+j0]*fx*(1-fy) + arr[i0*G+j1]*(1-fx)*fy + arr[i1*G+j1]*fx*fy;
}

function sdfGrad(sdf, gi, gj) {
  const e = 0.5;
  const dx = bilinear(sdf, Math.min(gi+e,255), gj) - bilinear(sdf, Math.max(gi-e,0), gj);
  const dy = bilinear(sdf, gi, Math.min(gj+e,255)) - bilinear(sdf, gi, Math.max(gj-e,0));
  const len = Math.sqrt(dx*dx + dy*dy) || 1;
  return [dx/len, dy/len];
}

// ============================================================
// Canvas setup — responsive
// ============================================================
let RW = 1024, RH = 682;
const heatCanvas = document.getElementById('c');
const heatCtx = heatCanvas.getContext('2d');
const partCanvas = document.getElementById('particles');
const partCtx = partCanvas.getContext('2d');
heatCanvas.width = RW; heatCanvas.height = RH;
partCanvas.width = RW; partCanvas.height = RH;

function resizeCanvas() {
  const w = window.innerWidth, h = window.innerHeight;
  const aspect = 3 / 2;
  let cw, ch;
  if (w / h > aspect) { ch = h; cw = Math.round(h * aspect); }
  else { cw = w; ch = Math.round(w / aspect); }
  cw = Math.max(cw, 256); ch = Math.max(ch, 170);
  if (cw !== RW || ch !== RH) {
    RW = cw; RH = ch;
    heatCanvas.width = RW; heatCanvas.height = RH;
    partCanvas.width = RW; partCanvas.height = RH;
  }
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// ============================================================
// Domain constants (matching web decode.js)
// ============================================================
const X_MIN = -1, X_MAX = 2, Y_MIN = -1, Y_MAX = 1;

function physToGrid(x, y) {
  return [(x - X_MIN) / (X_MAX - X_MIN) * 255, (1 - (y - Y_MIN) / (Y_MAX - Y_MIN)) * 255];
}

// ============================================================
// State
// ============================================================
let currentField = null, currentSdf = null, currentVmin = 0, currentVmax = 1, currentCmap = 'coolwarm';
let currentVx = null, currentVy = null;
let prevField = null;
let transProgress = 1;
let showParticles = true;
const blendBuf = new Float32Array(G * G);  // reused transition buffer (no per-frame alloc)

// ============================================================
// Particle system
// ============================================================
const PARTICLE_COUNT = 800;
const MAX_TRAIL = 12;
const PARTICLE_SPEED = 0.4;
const MAX_AGE = 120;

function spawnParticle() {
  return { x: X_MIN + Math.random() * 3, y: Y_MIN + Math.random() * 2, trail: [], age: Math.floor(Math.random() * MAX_AGE) };
}
const particles = Array.from({ length: PARTICLE_COUNT }, spawnParticle);

// ============================================================
// Heatmap render
// ============================================================
function renderHeatmap(field, sdf, vmin, vmax, cmapName) {
  const imageData = heatCtx.createImageData(RW, RH);
  const data = imageData.data;
  const lut = getLUT(cmapName);
  const range = vmax - vmin || 1;
  const invRange = 1 / range;
  const maxI = RW - 1, maxJ = RH - 1;
  const lightX = -0.4, lightY = 0.7;
  const sdfTh = 0.015;

  for (let py = 0; py < RH; py++) {
    const gj = (1 - py / maxJ) * 255;
    for (let px = 0; px < RW; px++) {
      const gi = (px / maxI) * 255;
      const val = bilinear(field, gi, gj);
      let norm = (val - vmin) * invRange;
      if (norm < 0) norm = 0; if (norm > 1) norm = 1;
      const li = (norm * 255 + 0.5 | 0) * 4;
      const pi = (py * RW + px) * 4;
      let r = lut[li], g = lut[li+1], b = lut[li+2];

      if (sdf) {
        const sv = bilinear(sdf, gi, gj);
        if (sv < sdfTh) {
          const depth = Math.min(1, (sdfTh - sv) / (sdfTh * 8));
          const [nx, ny] = sdfGrad(sdf, gi, gj);
          const diff = Math.max(0, nx*lightX + ny*lightY);
          const spec = Math.pow(diff, 12) * 0.6;
          const st = Math.max(0, ny) * 0.35;
          r = Math.round(Math.min(255, 32 + diff*45 + st*55 + spec*180));
          g = Math.round(Math.min(255, 36 + diff*50 + st*60 + spec*195));
          b = Math.round(Math.min(255, 48 + diff*60 + st*70 + spec*220));
          const cd = 1 - depth * 0.15;
          r = Math.round(r*cd); g = Math.round(g*cd); b = Math.round(b*cd);
        } else if (sv < sdfTh + 0.012) {
          const ed = (sv - sdfTh) / 0.012;
          const t = ed*ed*(3-2*ed);
          const [nx, ny] = sdfGrad(sdf, gi, gj);
          const diff = Math.max(0, nx*lightX + ny*lightY);
          const br = Math.round(32 + diff*45 + Math.max(0,ny)*20);
          const bg = Math.round(36 + diff*50 + Math.max(0,ny)*22);
          const bb = Math.round(48 + diff*60 + Math.max(0,ny)*25);
          const ss = (1-t)*0.3;
          r = Math.round(br*(1-t) + r*t*(1-ss));
          g = Math.round(bg*(1-t) + g*t*(1-ss));
          b = Math.round(bb*(1-t) + b*t*(1-ss));
        }
      }
      data[pi] = r; data[pi+1] = g; data[pi+2] = b; data[pi+3] = 255;
    }
  }
  heatCtx.putImageData(imageData, 0, 0);
}

// ============================================================
// Animation loop — transitions + particles
// ============================================================
let animId = null;

function animLoop() {
  // Smooth transition
  if (transProgress < 1 && prevField && currentField) {
    transProgress = Math.min(1, transProgress + 0.04);
    const blended = blendBuf;
    const t = transProgress;
    for (let k = 0; k < blended.length; k++) {
      blended[k] = prevField[k] * (1 - t) + currentField[k] * t;
    }
    renderHeatmap(blended, currentSdf, currentVmin, currentVmax, currentCmap);
  }

  // Particles
  if (showParticles && currentVx && currentVy) {
    partCtx.clearRect(0, 0, RW, RH);

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const [gi, gj] = physToGrid(p.x, p.y);
      const vx = bilinear(currentVx, Math.min(Math.max(gi,0),255), Math.min(Math.max(gj,0),255));
      const vy = bilinear(currentVy, Math.min(Math.max(gi,0),255), Math.min(Math.max(gj,0),255));

      p.trail.push({ x: p.x, y: p.y });
      if (p.trail.length > MAX_TRAIL) p.trail.shift();

      const speed = Math.sqrt(vx * vx + vy * vy);
      const dt = PARTICLE_SPEED / (speed + 0.01);
      p.x += vx * dt * 0.015;
      p.y += vy * dt * 0.015;
      p.age++;

      const [si, sj] = physToGrid(p.x, p.y);
      const sv = currentSdf ? bilinear(currentSdf, Math.min(Math.max(si,0),255), Math.min(Math.max(sj,0),255)) : 1;

      if (p.x < X_MIN || p.x > X_MAX || p.y < Y_MIN || p.y > Y_MAX || p.age > MAX_AGE || sv < 0.02) {
        Object.assign(p, spawnParticle());
        continue;
      }

      if (p.trail.length > 1) {
        partCtx.beginPath();
        const t0 = p.trail[0];
        partCtx.moveTo(((t0.x - X_MIN) / 3) * RW, (1 - (t0.y - Y_MIN) / 2) * RH);
        for (let t = 1; t < p.trail.length; t++) {
          const tp = p.trail[t];
          partCtx.lineTo(((tp.x - X_MIN) / 3) * RW, (1 - (tp.y - Y_MIN) / 2) * RH);
        }
        partCtx.strokeStyle = 'rgba(255,255,255,' + Math.min(0.6, speed * 2) + ')';
        partCtx.lineWidth = 0.8;
        partCtx.stroke();
      }
    }
  } else {
    partCtx.clearRect(0, 0, RW, RH);
  }

  animId = requestAnimationFrame(animLoop);
}
animId = requestAnimationFrame(animLoop);

// ============================================================
// Touch tooltips
// ============================================================
const tooltip = document.getElementById('tooltip');
const wrap = document.getElementById('wrap');

function handleTouch(e) {
  e.preventDefault();
  if (!currentField) return;
  const touch = e.touches[0];
  const rect = heatCanvas.getBoundingClientRect();
  const px = touch.clientX - rect.left;
  const py = touch.clientY - rect.top;
  const physX = X_MIN + (px / rect.width) * (X_MAX - X_MIN);
  const physY = Y_MAX - (py / rect.height) * (Y_MAX - Y_MIN);
  const [gi, gj] = physToGrid(physX, physY);
  const i = Math.round(gi), j = Math.round(gj);
  if (i >= 0 && i < 256 && j >= 0 && j < 256) {
    const val = currentField[i * 256 + j];
    const sv = currentSdf ? currentSdf[i * 256 + j] : 1;
    const inside = sv < 0.015;
    tooltip.innerHTML = 'x: ' + physX.toFixed(3) + ', y: ' + physY.toFixed(3) + '<br>' +
      (inside ? '<span class="airfoil">Inside airfoil</span>' : '<span class="val">Value: ' + val.toFixed(5) + '</span>');
    tooltip.className = 'visible';
    tooltip.style.left = Math.min(px + 12, rect.width - 140) + 'px';
    tooltip.style.top = Math.max(0, py - 40) + 'px';
  }
}
heatCanvas.addEventListener('touchstart', handleTouch, { passive: false });
heatCanvas.addEventListener('touchmove', handleTouch, { passive: false });
heatCanvas.addEventListener('touchend', () => { tooltip.className = ''; }, { passive: true });

// ============================================================
// Host messaging
// ============================================================
function postToHost(data) {
  const msg = JSON.stringify(data);
  if (window.ReactNativeWebView) window.ReactNativeWebView.postMessage(msg);
  else window.parent.postMessage(msg, '*');
}

function b64ToF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function onHostMessage(e) {
  try {
    const msg = JSON.parse(e.data);
    if (msg.type === 'render') {
      const newField = b64ToF32(msg.field);
      const sdf = msg.sdf ? b64ToF32(msg.sdf) : null;
      if (msg.vx) currentVx = b64ToF32(msg.vx);
      if (msg.vy) currentVy = b64ToF32(msg.vy);

      if (currentField) {
        prevField = currentField;
        transProgress = 0;
      }
      currentField = newField;
      currentSdf = sdf;
      currentVmin = msg.vmin;
      currentVmax = msg.vmax;
      currentCmap = msg.cmap || 'coolwarm';

      if (!prevField) {
        renderHeatmap(currentField, currentSdf, currentVmin, currentVmax, currentCmap);
      }
      postToHost({ type: 'rendered' });
    } else if (msg.type === 'config') {
      if (typeof msg.showParticles === 'boolean') showParticles = msg.showParticles;
    }
  } catch (err) {
    postToHost({ type: 'error', msg: err.message });
  }
}

// react-native-webview delivers ref.postMessage() on 'document' (Android) but
// 'window' (iOS); web iframe postMessage arrives on 'window'. Listen on both.
document.addEventListener('message', onHostMessage);
window.addEventListener('message', onHostMessage);

postToHost({ type: 'ready' });
</script>
</body>
</html>
`;

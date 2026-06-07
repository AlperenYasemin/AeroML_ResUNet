/**
 * Scientific Colormaps for aerodynamic field visualization.
 * Multi-stop linear interpolation between key color values.
 */

// --- Colormap Definitions (key stops: [position, R, G, B]) ---

const COOLWARM_STOPS = [
  [0.0,  59,  76, 192],
  [0.1,  77, 105, 220],
  [0.2, 107, 137, 243],
  [0.3, 141, 167, 254],
  [0.4, 177, 199, 252],
  [0.5, 221, 221, 221],
  [0.6, 243, 190, 170],
  [0.7, 245, 156, 125],
  [0.8, 235, 115,  82],
  [0.9, 210,  67,  43],
  [1.0, 180,   4,  38],
];

const VIRIDIS_STOPS = [
  [0.0,   68,   1,  84],
  [0.1,   72,  36, 117],
  [0.2,   64,  67, 135],
  [0.3,   52,  94, 141],
  [0.4,   41, 120, 142],
  [0.5,   33, 145, 140],
  [0.6,   40, 174, 128],
  [0.7,   68, 199, 101],
  [0.8,  122, 220,  62],
  [0.9,  189, 237,  27],
  [1.0,  253, 231,  37],
];

const INFERNO_STOPS = [
  [0.0,    0,   0,   4],
  [0.1,   40,  11,  84],
  [0.2,   87,  16, 110],
  [0.3,  132,  32, 107],
  [0.4,  169,  54,  89],
  [0.5,  203,  77,  67],
  [0.6,  227, 105,  43],
  [0.7,  244, 139,  18],
  [0.8,  249, 178,  10],
  [0.9,  241, 220,  72],
  [1.0,  252, 255, 164],
];

const TURBO_STOPS = [
  [0.00,  48,  18,  59],
  [0.07,  64,  57, 168],
  [0.14,  65, 101, 228],
  [0.21,  38, 147, 252],
  [0.28,  14, 189, 228],
  [0.35,  21, 222, 191],
  [0.42,  72, 244, 137],
  [0.49, 136, 253,  75],
  [0.56, 190, 248,  38],
  [0.63, 228, 228,  24],
  [0.70, 251, 196,  20],
  [0.77, 252, 157,  18],
  [0.84, 239, 111,  12],
  [0.91, 209,  60,   5],
  [1.00, 122,   4,   3],
];

const COLORMAPS = {
  coolwarm: COOLWARM_STOPS,
  viridis: VIRIDIS_STOPS,
  inferno: INFERNO_STOPS,
  turbo: TURBO_STOPS,
};

/**
 * Build a 256-entry lookup table from colormap stops.
 * Returns Uint8Array of length 256*4 (RGBA).
 */
function buildLUT(stops) {
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    // Find the two stops surrounding t
    let lo = 0;
    for (let s = 1; s < stops.length; s++) {
      if (stops[s][0] >= t) { lo = s - 1; break; }
      if (s === stops.length - 1) lo = s - 1;
    }
    const hi = Math.min(lo + 1, stops.length - 1);
    const range = stops[hi][0] - stops[lo][0];
    const frac = range > 0 ? (t - stops[lo][0]) / range : 0;

    lut[i * 4 + 0] = Math.round(stops[lo][1] + (stops[hi][1] - stops[lo][1]) * frac);
    lut[i * 4 + 1] = Math.round(stops[lo][2] + (stops[hi][2] - stops[lo][2]) * frac);
    lut[i * 4 + 2] = Math.round(stops[lo][3] + (stops[hi][3] - stops[lo][3]) * frac);
    lut[i * 4 + 3] = 255;
  }
  return lut;
}

// Pre-build all LUTs
const LUT_CACHE = {};
for (const name of Object.keys(COLORMAPS)) {
  LUT_CACHE[name] = buildLUT(COLORMAPS[name]);
}

/**
 * Map a normalized value [0,1] to an RGBA color using a named colormap.
 * Returns [r, g, b, a].
 */
export function colormapRGBA(value, cmapName = 'coolwarm') {
  const lut = LUT_CACHE[cmapName] || LUT_CACHE.coolwarm;
  const idx = Math.max(0, Math.min(255, Math.round(value * 255))) * 4;
  return [lut[idx], lut[idx + 1], lut[idx + 2], lut[idx + 3]];
}

/**
 * Get the pre-built LUT for a colormap.
 */
export function getLUT(cmapName = 'coolwarm') {
  return LUT_CACHE[cmapName] || LUT_CACHE.coolwarm;
}

/**
 * Render a field (Float32Array, 256×256) into a high-resolution ImageData.
 * Uses bilinear interpolation to upscale from the 256×256 grid to the
 * target render resolution, eliminating visible pixelation.
 *
 * Airfoil interior is rendered as a realistic solid body with:
 *   - Carbon composite surface gradient (lighter upper, darker lower)
 *   - Specular highlight along the upper leading edge
 *   - Smooth anti-aliased boundary with ambient occlusion shadow
 *
 * field is row-major: field[i*256+j], i=x-index, j=y-index.
 * renderW/renderH default to 1536×1024 (3:2 matching the domain).
 */
export const RENDER_W = 1536;
export const RENDER_H = 1024;

// Inline bilinear sampler (avoids function call overhead in hot loop)
function bilinear(arr, gi, gj) {
  const G = 256;
  const i0 = gi | 0;  // fast floor
  const j0 = gj | 0;
  const i1 = i0 < 255 ? i0 + 1 : 255;
  const j1 = j0 < 255 ? j0 + 1 : 255;
  const fx = gi - i0;
  const fy = gj - j0;
  return (
    arr[i0 * G + j0] * (1 - fx) * (1 - fy) +
    arr[i1 * G + j0] * fx * (1 - fy) +
    arr[i0 * G + j1] * (1 - fx) * fy +
    arr[i1 * G + j1] * fx * fy
  );
}

// Compute the SDF gradient (surface normal) via central differences
function sdfGradient(sdf, gi, gj) {
  const eps = 0.5;
  const dfdx = bilinear(sdf, Math.min(gi + eps, 255), gj) - bilinear(sdf, Math.max(gi - eps, 0), gj);
  const dfdy = bilinear(sdf, gi, Math.min(gj + eps, 255)) - bilinear(sdf, gi, Math.max(gj - eps, 0));
  const len = Math.sqrt(dfdx * dfdx + dfdy * dfdy) || 1;
  return [dfdx / len, dfdy / len];
}

export function renderFieldToImageData(
  field, vmin, vmax, cmapName, sdf,
  sdfThreshold = 0.015,
  renderW = RENDER_W, renderH = RENDER_H
) {
  const imageData = new ImageData(renderW, renderH);
  const data = imageData.data;
  const lut = getLUT(cmapName);
  const range = vmax - vmin || 1;
  const invRange = 1 / range;
  const maxI = renderW - 1;
  const maxJ = renderH - 1;

  // Light direction (upper-left, slightly forward) for specular highlight
  const lightX = -0.4, lightY = 0.7;

  for (let py = 0; py < renderH; py++) {
    const gj = (1 - py / maxJ) * 255;

    for (let px = 0; px < renderW; px++) {
      const gi = (px / maxI) * 255;

      // Bilinear interpolation of the field value
      const val = bilinear(field, gi, gj);

      // Map to colormap
      let norm = (val - vmin) * invRange;
      if (norm < 0) norm = 0;
      if (norm > 1) norm = 1;
      const lutIdx = (norm * 255 + 0.5 | 0) * 4;
      const pixIdx = (py * renderW + px) * 4;

      let r = lut[lutIdx];
      let g = lut[lutIdx + 1];
      let b = lut[lutIdx + 2];

      // --- Airfoil rendering ---
      if (sdf) {
        const sdfVal = bilinear(sdf, gi, gj);

        if (sdfVal < sdfThreshold) {
          // =========================================
          // INSIDE AIRFOIL — realistic solid body
          // =========================================

          // How deep inside (0 = edge, 1 = deep core)
          const depth = Math.min(1, (sdfThreshold - sdfVal) / (sdfThreshold * 8));

          // Surface normal for lighting
          const [nx, ny] = sdfGradient(sdf, gi, gj);

          // Diffuse lighting: dot(normal, light_dir)
          const diffuse = Math.max(0, nx * lightX + ny * lightY);

          // Specular: Blinn-Phong style
          const specular = Math.pow(diffuse, 12) * 0.6;

          // Base color: dark carbon composite (charcoal-blue)
          const baseR = 32, baseG = 36, baseB = 48;

          // Gradient: lighter on the upper surface (ny > 0 = top)
          const surfaceTint = Math.max(0, ny) * 0.35;

          // Combine: base + diffuse lighting + surface tint + specular
          r = Math.round(Math.min(255, baseR + diffuse * 45 + surfaceTint * 55 + specular * 180));
          g = Math.round(Math.min(255, baseG + diffuse * 50 + surfaceTint * 60 + specular * 195));
          b = Math.round(Math.min(255, baseB + diffuse * 60 + surfaceTint * 70 + specular * 220));

          // Inner core darkening for 3D volume feel
          const coreDarken = 1 - depth * 0.15;
          r = Math.round(r * coreDarken);
          g = Math.round(g * coreDarken);
          b = Math.round(b * coreDarken);

        } else if (sdfVal < sdfThreshold + 0.012) {
          // =========================================
          // NEAR EDGE — anti-aliased boundary + shadow
          // =========================================
          const edgeDist = (sdfVal - sdfThreshold) / 0.012;
          // Smoothstep for anti-aliasing
          const t = edgeDist * edgeDist * (3 - 2 * edgeDist);

          // Airfoil body color at the edge (use normal for lighting)
          const [nx, ny] = sdfGradient(sdf, gi, gj);
          const diffuse = Math.max(0, nx * lightX + ny * lightY);
          const bodyR = Math.round(32 + diffuse * 45 + Math.max(0, ny) * 20);
          const bodyG = Math.round(36 + diffuse * 50 + Math.max(0, ny) * 22);
          const bodyB = Math.round(48 + diffuse * 60 + Math.max(0, ny) * 25);

          // Ambient occlusion shadow just outside the surface
          const shadowFade = (1 - t);
          const shadowStrength = shadowFade * 0.3;

          // Blend: field ← airfoil body (anti-aliased) + shadow
          r = Math.round(bodyR * (1 - t) + r * t * (1 - shadowStrength));
          g = Math.round(bodyG * (1 - t) + g * t * (1 - shadowStrength));
          b = Math.round(bodyB * (1 - t) + b * t * (1 - shadowStrength));
        }
      }

      data[pixIdx]     = r;
      data[pixIdx + 1] = g;
      data[pixIdx + 2] = b;
      data[pixIdx + 3] = 255;
    }
  }
  return imageData;
}

export const COLORMAP_NAMES = Object.keys(COLORMAPS);


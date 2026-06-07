/**
 * Decode base64-encoded binary float32 arrays from the API.
 */

/**
 * Convert a base64 string to a Float32Array.
 * The server sends raw float32 bytes encoded as base64.
 */
export function base64ToFloat32Array(base64String) {
  const binaryString = atob(base64String);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

/**
 * Bilinear interpolation lookup in a 256×256 field.
 * field is row-major: field[i*256+j], i=x-index, j=y-index.
 * x,y are in grid coordinates (0–255).
 */
export function sampleField(field, x, y) {
  const W = 256, H = 256;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, W - 1), y1 = Math.min(y0 + 1, H - 1);
  const fx = x - x0, fy = y - y0;

  const clampX0 = Math.max(0, Math.min(W - 1, x0));
  const clampY0 = Math.max(0, Math.min(H - 1, y0));
  const clampX1 = Math.max(0, Math.min(W - 1, x1));
  const clampY1 = Math.max(0, Math.min(H - 1, y1));

  const v00 = field[clampX0 * H + clampY0];
  const v10 = field[clampX1 * H + clampY0];
  const v01 = field[clampX0 * H + clampY1];
  const v11 = field[clampX1 * H + clampY1];

  return (
    v00 * (1 - fx) * (1 - fy) +
    v10 * fx * (1 - fy) +
    v01 * (1 - fx) * fy +
    v11 * fx * fy
  );
}

// Physical domain constants
export const X_MIN = -1.0, X_MAX = 2.0;
export const Y_MIN = -1.0, Y_MAX = 1.0;

/** Convert physical (x,y) to grid indices (i,j). */
export function physToGrid(x, y) {
  const i = ((x - X_MIN) / (X_MAX - X_MIN)) * 255;
  const j = ((y - Y_MIN) / (Y_MAX - Y_MIN)) * 255;
  return [i, j];
}

/** Convert canvas pixel (px,py) to physical (x,y). canvasW/H = display size. */
export function canvasToPhys(px, py, canvasW, canvasH) {
  const x = X_MIN + (px / canvasW) * (X_MAX - X_MIN);
  const y = Y_MAX - (py / canvasH) * (Y_MAX - Y_MIN); // flip y
  return [x, y];
}

import { Platform } from 'react-native';

const CLOUD_URL = 'https://aeroml-gpu-319266263960.us-central1.run.app';

const getDefaultBaseUrl = (): string => {
  return CLOUD_URL;
};

let API_BASE = getDefaultBaseUrl();

export const setApiBase = (url: string) => {
  API_BASE = url.replace(/\/$/, '');
};

export const getApiBase = () => API_BASE;

export interface SampleInfo {
  idx: number;
  thickness: number;
  camber: number;
  velocity: number;
  aoa: number;
}

export interface RawPrediction {
  width: number;
  height: number;
  pressure: string;
  velocity_x: string;
  velocity_y: string;
  sdf: string;
  pressure_min: number;
  pressure_max: number;
  velocity_x_min: number;
  velocity_x_max: number;
  velocity_y_min: number;
  velocity_y_max: number;
  inference_time_ms: number;
  device: string;
}

export interface HealthStatus {
  status: string;
  device: string;
  model_loaded: boolean;
  available_samples: number[];
}

export const checkHealth = async (): Promise<HealthStatus | null> => {
  // NOTE: AbortSignal.timeout() is NOT available in React Native's bundled
  // AbortSignal polyfill (abort-controller pkg) — it throws in release builds
  // on Hermes. Use a manual AbortController + setTimeout instead.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 3000);
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    if (res.ok) return await res.json();
    return null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
};

export const fetchCatalog = async (): Promise<SampleInfo[]> => {
  try {
    const res = await fetch(`${API_BASE}/samples/catalog`);
    if (res.ok) {
      const data = await res.json();
      return data.samples || [];
    }
    return [];
  } catch {
    return [];
  }
};

let activeController: AbortController | null = null;

export const predictRaw = async (params: {
  ux_in: number;
  uy_in: number;
  airfoil?: string;
  sample_idx?: number;
}): Promise<RawPrediction | null> => {
  // Cancel any in-flight prediction
  if (activeController) activeController.abort();
  activeController = new AbortController();

  try {
    const res = await fetch(`${API_BASE}/predict/raw`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
      signal: activeController.signal,
    });
    activeController = null;
    if (res.ok) return await res.json();
    return null;
  } catch (err: any) {
    if (err?.name === 'AbortError') return null;
    // Retry once on transient failure
    try {
      const res = await fetch(`${API_BASE}/predict/raw`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      if (res.ok) return await res.json();
    } catch {}
    return null;
  }
};

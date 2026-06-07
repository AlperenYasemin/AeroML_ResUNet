import { useState, useEffect, useRef, useCallback } from 'react';
import {
  checkHealth,
  fetchCatalog,
  predictRaw,
  getApiBase,
  setApiBase,
  SampleInfo,
  RawPrediction,
} from '../api/client';
import { FieldKey, FIELDS } from '../constants/theme';

export type ColormapName = 'coolwarm' | 'inferno' | 'viridis' | 'plasma';
export const COLORMAP_OPTIONS: ColormapName[] = ['coolwarm', 'inferno', 'viridis', 'plasma'];

const DEBOUNCE_MS = 350;

export interface PredictionState {
  velocity: number;
  aoa: number;
  camber: number;
  camberPos: number;
  thickness: number;
  activeField: FieldKey;
  geometrySource: 'training' | 'analytical';
  rawData: RawPrediction | null;
  inferenceTime: number | null;
  serverOnline: boolean | null;
  deviceName: string;
  catalog: SampleInfo[];
  matchedSample: SampleInfo | null;
  predictionCount: number;
  isLoading: boolean;
  error: string | null;
  showSettings: boolean;
  serverUrl: string;
}

export function usePrediction() {
  const [velocity, setVelocity] = useState(40);
  const [aoa, setAoa] = useState(5);
  const [camber, setCamber] = useState(0);
  const [camberPos, setCamberPos] = useState(4);
  const [thickness, setThickness] = useState(12);
  const [activeField, setActiveField] = useState<FieldKey>('pressure');
  const [geometrySource, setGeometrySource] = useState<'training' | 'analytical'>('training');
  const [colormaps, setColormaps] = useState<Record<FieldKey, ColormapName>>({
    pressure: 'coolwarm',
    velocity_x: 'inferno',
    velocity_y: 'viridis',
  });
  const setColormapForField = (field: FieldKey, cmap: ColormapName) => {
    setColormaps(prev => ({ ...prev, [field]: cmap }));
  };
  const [showParticles, setShowParticles] = useState(true);

  const [rawData, setRawData] = useState<RawPrediction | null>(null);
  const [inferenceTime, setInferenceTime] = useState<number | null>(null);
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);
  const [deviceName, setDeviceName] = useState('');
  const [catalog, setCatalog] = useState<SampleInfo[]>([]);
  const [matchedSample, setMatchedSample] = useState<SampleInfo | null>(null);
  const [predictionCount, setPredictionCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [serverUrl, setServerUrl] = useState(getApiBase());

  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const isFirstRender = useRef(true);
  const activeFieldRef = useRef<FieldKey>(activeField);
  useEffect(() => { activeFieldRef.current = activeField; }, [activeField]);

  const alphaRad = (aoa * Math.PI) / 180;
  const uxIn = velocity * Math.cos(alphaRad);
  const uyIn = velocity * Math.sin(alphaRad);
  const nacaCode = `${camber}${camber === 0 ? 0 : camberPos}${thickness.toString().padStart(2, '0')}`;

  // Health check every 15s
  useEffect(() => {
    const check = async () => {
      const health = await checkHealth();
      setServerOnline(!!health);
      if (health) setDeviceName(health.device);
    };
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  // Fetch catalog
  useEffect(() => {
    fetchCatalog().then(setCatalog);
  }, []);

  // Match closest training sample
  useEffect(() => {
    if (catalog.length === 0 || geometrySource !== 'training') {
      setMatchedSample(null);
      return;
    }
    let best: SampleInfo | null = null;
    let bestScore = Infinity;
    for (const s of catalog) {
      const score = Math.abs(s.thickness - thickness) * 2 + Math.abs(s.camber - camber) * 3;
      if (score < bestScore) { bestScore = score; best = s; }
    }
    setMatchedSample(best);
  }, [catalog, thickness, camber, geometrySource]);

  const runPrediction = useCallback(async () => {
    if (!serverOnline) return;
    setIsLoading(true);
    setError(null);
    const body: any = { ux_in: uxIn, uy_in: uyIn };
    if (geometrySource === 'training' && matchedSample) {
      body.sample_idx = matchedSample.idx;
    } else {
      body.airfoil = `naca${nacaCode}`;
    }
    const result = await predictRaw(body);
    if (result) {
      setRawData(result);
      setInferenceTime(result.inference_time_ms);
      setDeviceName(result.device);
      setPredictionCount(c => c + 1);
    } else {
      setError('Prediction failed — check server connection.');
    }
    setIsLoading(false);
  }, [uxIn, uyIn, nacaCode, matchedSample, geometrySource, serverOnline]);

  // Auto-predict on parameter change with debounce
  useEffect(() => {
    if (!serverOnline) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(
      () => runPrediction(),
      isFirstRender.current ? 800 : DEBOUNCE_MS,
    );
    if (isFirstRender.current) isFirstRender.current = false;
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [velocity, aoa, camber, camberPos, thickness, geometrySource, runPrediction, serverOnline]);

  const saveServerUrl = () => {
    setApiBase(serverUrl);
    setShowSettings(false);
    checkHealth().then(h => { setServerOnline(!!h); if (h) setDeviceName(h.device); });
    fetchCatalog().then(setCatalog);
  };

  return {
    velocity, setVelocity,
    aoa, setAoa,
    camber, setCamber,
    camberPos, setCamberPos,
    thickness, setThickness,
    activeField, setActiveField, activeFieldRef,
    geometrySource, setGeometrySource,
    rawData, inferenceTime, serverOnline, deviceName,
    catalog, matchedSample,
    predictionCount, isLoading, error, setError,
    showSettings, setShowSettings,
    serverUrl, setServerUrl, saveServerUrl,
    uxIn, uyIn, nacaCode,
    colormaps, setColormapForField,
    showParticles, setShowParticles,
  };
}

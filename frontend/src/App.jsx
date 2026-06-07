import { useState, useEffect, useCallback, useRef } from 'react'
import FieldRenderer from './components/FieldRenderer'
import { base64ToFloat32Array } from './utils/decode'
import { COLORMAP_NAMES } from './utils/colormap'
import './App.css'

// API base URL: env var for Docker/Cloud Run, falls back to cloud URL, then localhost for dev.
const API_BASE = import.meta.env.VITE_API_URL ?? 'https://aeroml-gpu-319266263960.us-central1.run.app'
const DEBOUNCE_MS = 250

const FIELDS = [
  { key: 'pressure', label: 'Pressure (p)', cmap: 'coolwarm' },
  { key: 'velocity_x', label: 'Velocity Ux', cmap: 'inferno' },
  { key: 'velocity_y', label: 'Velocity Uy', cmap: 'viridis' },
]

// Common NACA presets for quick selection
const NACA_PRESETS = [
  { m: 0, p: 0, t: 12, label: 'Symmetric (0012)' },
  { m: 2, p: 4, t: 12, label: 'Standard (2412)' },
  { m: 4, p: 4, t: 12, label: 'Medium Camber (4412)' },
  { m: 4, p: 4, t: 15, label: 'Thick Camber (4415)' },
  { m: 0, p: 0, t: 8,  label: 'Thin Symmetric (0008)' },
  { m: 6, p: 4, t: 12, label: 'High Camber (6412)' },
]

function App() {
  // Controls
  const [velocity, setVelocity] = useState(40)
  const [angleOfAttack, setAngleOfAttack] = useState(5)
  // NACA 4-digit parameters
  const [camber, setCamber] = useState(0)        // m: max camber (0–6% chord)
  const [camberPos, setCamberPos] = useState(4)   // p: camber position (1–7 × 10% chord)
  const [thickness, setThickness] = useState(12)  // t: max thickness (6–24% chord)
  const [showParticles, setShowParticles] = useState(true)
  const [activeField, setActiveField] = useState('pressure')
  const [colormaps, setColormaps] = useState({
    pressure: 'coolwarm',
    velocity_x: 'inferno',
    velocity_y: 'viridis',
  })

  // Geometry source: 'training' = real training data, 'analytical' = NACA SDF generator
  const [geometrySource, setGeometrySource] = useState('training')
  const [sampleCatalog, setSampleCatalog] = useState([])
  const [matchedSample, setMatchedSample] = useState(null)

  // Data
  const [rawData, setRawData] = useState(null)
  const [inferenceTime, setInferenceTime] = useState(null)
  const [device, setDevice] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [serverOnline, setServerOnline] = useState(null)
  const [error, setError] = useState(null)
  const [predictionCount, setPredictionCount] = useState(0)

  const debounceTimer = useRef(null)
  const isFirstRender = useRef(true)

  // Computed
  const alphaRad = (angleOfAttack * Math.PI) / 180
  const uxIn = velocity * Math.cos(alphaRad)
  const uyIn = velocity * Math.sin(alphaRad)

  // Construct NACA code from slider values
  const nacaCode = `${camber}${camber === 0 ? 0 : camberPos}${thickness.toString().padStart(2, '0')}`
  const airfoilCode = `naca${nacaCode}`

  // Health check + fetch sample catalog
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`)
        if (res.ok) {
          const data = await res.json()
          setServerOnline(true)
          setDevice(data.device)
        } else setServerOnline(false)
      } catch { setServerOnline(false) }
    }
    check()
    const interval = setInterval(check, 30000)
    return () => clearInterval(interval)
  }, [])

  // Fetch sample catalog on mount
  useEffect(() => {
    const fetchCatalog = async () => {
      try {
        const res = await fetch(`${API_BASE}/samples/catalog`)
        if (res.ok) {
          const data = await res.json()
          setSampleCatalog(data.samples || [])
        }
      } catch { /* catalog not available */ }
    }
    fetchCatalog()
  }, [])

  // Find closest matching training sample based on slider values
  useEffect(() => {
    if (sampleCatalog.length === 0 || geometrySource !== 'training') {
      setMatchedSample(null)
      return
    }
    // Score each sample by how close its properties match the sliders
    let bestSample = null
    let bestScore = Infinity
    for (const s of sampleCatalog) {
      // Weight: thickness match is most important, then camber
      const thicknessErr = Math.abs(s.thickness - thickness) * 2
      const camberErr = Math.abs(s.camber - camber) * 3
      const score = thicknessErr + camberErr
      if (score < bestScore) {
        bestScore = score
        bestSample = s
      }
    }
    setMatchedSample(bestSample)
  }, [sampleCatalog, thickness, camber, camberPos, geometrySource])

  // Prediction (raw data)
  const runPrediction = useCallback(async () => {
    if (!serverOnline) return
    setIsLoading(true)
    setError(null)
    try {
      const body = { ux_in: uxIn, uy_in: uyIn }

      if (geometrySource === 'training' && matchedSample) {
        // Use real training data SDF
        body.sample_idx = matchedSample.idx
      } else {
        // Use analytical NACA SDF
        body.airfoil = airfoilCode
      }

      const res = await fetch(`${API_BASE}/predict/raw`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)

      const data = await res.json()

      setRawData({
        pressure: base64ToFloat32Array(data.pressure),
        velocity_x: base64ToFloat32Array(data.velocity_x),
        velocity_y: base64ToFloat32Array(data.velocity_y),
        sdf: base64ToFloat32Array(data.sdf),
        ranges: {
          pressure: [data.pressure_min, data.pressure_max],
          velocity_x: [data.velocity_x_min, data.velocity_x_max],
          velocity_y: [data.velocity_y_min, data.velocity_y_max],
        },
      })
      setInferenceTime(data.inference_time_ms)
      setDevice(data.device)
      setPredictionCount(c => c + 1)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }, [uxIn, uyIn, airfoilCode, matchedSample, geometrySource, serverOnline])

  // Auto-predict on any parameter change (including first load)
  useEffect(() => {
    if (!serverOnline) return
    if (debounceTimer.current) clearTimeout(debounceTimer.current)
    debounceTimer.current = setTimeout(() => runPrediction(), isFirstRender.current ? 500 : DEBOUNCE_MS)
    if (isFirstRender.current) isFirstRender.current = false
    return () => { if (debounceTimer.current) clearTimeout(debounceTimer.current) }
  }, [velocity, angleOfAttack, camber, camberPos, thickness, geometrySource, runPrediction, serverOnline])

  const renderField = (fieldKey) => {
    const f = FIELDS.find(f => f.key === fieldKey)
    if (!rawData) return null
    const [vmin, vmax] = rawData.ranges[fieldKey]
    return (
      <FieldRenderer
        key={fieldKey}
        fieldData={rawData[fieldKey]}
        velocityX={rawData.velocity_x}
        velocityY={rawData.velocity_y}
        sdf={rawData.sdf}
        vmin={vmin}
        vmax={vmax}
        colormap={colormaps[fieldKey]}
        label={f.label}
        showParticles={showParticles}
        animating={true}
      />
    )
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header__brand">
          <div className="header__logo">A</div>
          <div>
            <div className="header__title">AeroML</div>
            <div className="header__subtitle">Real-Time Aerodynamic Prediction</div>
          </div>
        </div>
        <div className="header__status">
          {inferenceTime !== null && (
            <div className="inference-timer">
              <span className="inference-timer__dot" />
              {inferenceTime.toFixed(1)} ms
            </div>
          )}
          {predictionCount > 0 && (
            <div className="inference-timer" style={{ borderColor: 'rgba(99,102,241,0.2)' }}>
              #{predictionCount}
            </div>
          )}
          <div className={`status-badge ${
            serverOnline === null ? 'status-badge--loading' :
            serverOnline ? 'status-badge--online' : 'status-badge--offline'
          }`}>
            <span className="status-dot" />
            {serverOnline === null ? 'Connecting...' :
             serverOnline ? `${device?.toUpperCase() || 'Online'}` : 'Offline'}
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="main">
        {/* Sidebar */}
        <aside className="control-panel">
          <div className="panel-section">
            <div className="panel-section__title">Boundary Conditions</div>

            <div className="slider-group">
              <div className="slider-group__header">
                <span className="slider-group__label">Angle of Attack (α)</span>
                <span className="slider-group__value">{angleOfAttack.toFixed(1)}°</span>
              </div>
              <input id="aoa-slider" type="range" className="slider-track"
                min="-5" max="15" step="0.5" value={angleOfAttack}
                onChange={e => setAngleOfAttack(parseFloat(e.target.value))} />
              <div className="slider-range"><span>-5°</span><span>15°</span></div>
            </div>

            <div className="slider-group">
              <div className="slider-group__header">
                <span className="slider-group__label">Freestream Velocity (U∞)</span>
                <span className="slider-group__value">{velocity.toFixed(0)} m/s</span>
              </div>
              <input id="velocity-slider" type="range" className="slider-track"
                min="10" max="80" step="1" value={velocity}
                onChange={e => setVelocity(parseFloat(e.target.value))} />
              <div className="slider-range"><span>10 m/s</span><span>80 m/s</span></div>
            </div>
          </div>

          <div className="panel-section">
            <div className="panel-section__title">Airfoil Geometry</div>

            {/* Geometry Source Toggle */}
            <div className="source-toggle">
              <button
                className={`source-toggle__btn ${geometrySource === 'training' ? 'source-toggle__btn--active' : ''}`}
                onClick={() => setGeometrySource('training')}>
                <span className="source-toggle__icon">📊</span>
                Training Data
                <span className="source-toggle__badge">{sampleCatalog.length}</span>
              </button>
              <button
                className={`source-toggle__btn ${geometrySource === 'analytical' ? 'source-toggle__btn--active' : ''}`}
                onClick={() => setGeometrySource('analytical')}>
                <span className="source-toggle__icon">📐</span>
                Custom NACA
              </button>
            </div>

            {/* NACA code badge */}
            <div className="naca-badge">
              <span className="naca-badge__label">
                {geometrySource === 'training' ? 'Best Match' : 'Profile'}
              </span>
              <span className="naca-badge__code">
                {geometrySource === 'training' && matchedSample
                  ? `#${matchedSample.idx}`
                  : `NACA ${nacaCode}`}
              </span>
            </div>

            {/* Matched sample info (training mode) */}
            {geometrySource === 'training' && matchedSample && (
              <div className="match-info">
                <div className="match-info__row">
                  <span>Thickness</span>
                  <span className="match-info__value">{matchedSample.thickness.toFixed(1)}%</span>
                </div>
                <div className="match-info__row">
                  <span>Camber</span>
                  <span className="match-info__value">{matchedSample.camber.toFixed(1)}%</span>
                </div>
                <div className="match-info__row">
                  <span>Training AoA</span>
                  <span className="match-info__value">{matchedSample.aoa.toFixed(1)}°</span>
                </div>
                <div className="match-info__row">
                  <span>Training V</span>
                  <span className="match-info__value">{matchedSample.velocity.toFixed(0)} m/s</span>
                </div>
              </div>
            )}

            {/* Max Camber slider */}
            <div className="slider-group">
              <div className="slider-group__header">
                <span className="slider-group__label">Max Camber (m)</span>
                <span className="slider-group__value">{camber}%</span>
              </div>
              <input id="camber-slider" type="range" className="slider-track"
                min="0" max="6" step="1" value={camber}
                onChange={e => setCamber(parseInt(e.target.value))} />
              <div className="slider-range">
                <span>0% (symmetric)</span>
                <span>6%</span>
              </div>
            </div>

            {/* Camber Position slider */}
            <div className="slider-group" style={{ opacity: camber === 0 ? 0.35 : 1, pointerEvents: camber === 0 ? 'none' : 'auto' }}>
              <div className="slider-group__header">
                <span className="slider-group__label">Camber Position (p)</span>
                <span className="slider-group__value">{camberPos * 10}% chord</span>
              </div>
              <input id="camber-pos-slider" type="range" className="slider-track"
                min="1" max="7" step="1" value={camberPos}
                onChange={e => setCamberPos(parseInt(e.target.value))} />
              <div className="slider-range">
                <span>10%</span>
                <span>70%</span>
              </div>
            </div>

            {/* Thickness slider */}
            <div className="slider-group">
              <div className="slider-group__header">
                <span className="slider-group__label">Max Thickness (t)</span>
                <span className="slider-group__value">{thickness}%</span>
              </div>
              <input id="thickness-slider" type="range" className="slider-track"
                min="6" max="24" step="1" value={thickness}
                onChange={e => setThickness(parseInt(e.target.value))} />
              <div className="slider-range">
                <span>6% (thin)</span>
                <span>24% (thick)</span>
              </div>
            </div>

            {/* Quick presets */}
            <div className="slider-group" style={{ gap: '4px' }}>
              <span className="slider-group__label" style={{ fontSize: '0.75rem' }}>Quick Presets</span>
              <div className="preset-chips">
                {NACA_PRESETS.map(p => (
                  <button key={p.label}
                    className={`preset-chip ${camber === p.m && camberPos === p.p && thickness === p.t ? 'preset-chip--active' : ''}`}
                    onClick={() => { setCamber(p.m); setCamberPos(p.p); setThickness(p.t) }}>
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="panel-section">
            <div className="panel-section__title">Visualization</div>
            <label className="toggle-label">
              <input type="checkbox" checked={showParticles}
                onChange={e => setShowParticles(e.target.checked)}
                style={{ accentColor: 'var(--accent-primary)' }} />
              Show flow particles
            </label>
            {FIELDS.map(f => (
              <div key={f.key} className="slider-group" style={{ gap: '4px' }}>
                <span className="slider-group__label" style={{ fontSize: '0.75rem' }}>{f.label} colormap</span>
                <select className="airfoil-select" style={{ padding: '6px 10px', fontSize: '0.75rem' }}
                  value={colormaps[f.key]}
                  onChange={e => setColormaps(prev => ({ ...prev, [f.key]: e.target.value }))}>
                  {COLORMAP_NAMES.map(name => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </div>
            ))}
          </div>

          <div className="panel-section">
            <div className="panel-section__title">Computed Inputs</div>
            <div className="info-grid">
              <div className="info-card">
                <span className="info-card__label">Ux</span>
                <span className="info-card__value">
                  {uxIn.toFixed(1)}<span className="info-card__unit"> m/s</span>
                </span>
              </div>
              <div className="info-card">
                <span className="info-card__label">Uy</span>
                <span className="info-card__value">
                  {uyIn.toFixed(1)}<span className="info-card__unit"> m/s</span>
                </span>
              </div>
            </div>
          </div>



          {error && (
            <div className="error-banner">{error}</div>
          )}
        </aside>

        {/* Visualization */}
        <section className="viz-area">
          <div className="viz-header">
            <h1 className="viz-header__title">Aerodynamic Field Visualization</h1>
            <div className="field-tabs">
              {FIELDS.map(f => (
                <button key={f.key}
                  className={`field-tab ${activeField === f.key ? 'field-tab--active' : ''}`}
                  onClick={() => setActiveField(f.key)}>{f.label}</button>
              ))}
            </div>
          </div>

          {rawData ? (
            <div className="viz-single">
              {renderField(activeField)}
            </div>
          ) : (
            <div className="heatmap-container">
              {isLoading ? (
                <div className="spinner-overlay"><div className="spinner" /></div>
              ) : (
                <div className="heatmap-placeholder">
                  <div className="heatmap-placeholder__icon">🌊</div>
                  <div className="heatmap-placeholder__text">
                    {serverOnline === false ? (
                      <>Backend server is offline. Start it with:<br />
                      <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-secondary)' }}>
                        uvicorn src.main:app --port 8000
                      </code></>
                    ) : (
                      'Adjust parameters to begin — predictions run automatically.'
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App

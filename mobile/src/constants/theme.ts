import { Platform } from 'react-native';

export const COLORS = {
  bg: '#0f1117',
  bgSecondary: '#1a1d2e',
  bgCanvas: '#0a0c12',
  accent: '#6366f1',
  accentLight: '#818cf8',
  accentMuted: '#a78bfa',
  text: '#e2e8f0',
  textMuted: '#94a3b8',
  textDim: '#64748b',
  border: 'rgba(255,255,255,0.06)',
  borderAccent: 'rgba(99,102,241,0.2)',
  success: '#10b981',
  error: '#ef4444',
} as const;

export const FONTS = {
  mono: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
} as const;

export const FIELDS = [
  { key: 'pressure' as const, label: 'Pressure', cmap: 'coolwarm' },
  { key: 'velocity_x' as const, label: 'Velocity Ux', cmap: 'inferno' },
  { key: 'velocity_y' as const, label: 'Velocity Uy', cmap: 'viridis' },
];

export type FieldKey = typeof FIELDS[number]['key'];

export const NACA_PRESETS = [
  { m: 0, p: 0, t: 12, label: '0012' },
  { m: 2, p: 4, t: 12, label: '2412' },
  { m: 4, p: 4, t: 12, label: '4412' },
  { m: 4, p: 4, t: 15, label: '4415' },
  { m: 0, p: 0, t: 8, label: '0008' },
  { m: 6, p: 4, t: 12, label: '6412' },
];

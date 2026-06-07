import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import Slider from '@react-native-community/slider';
import { COLORS, FONTS, NACA_PRESETS } from '../constants/theme';
import { SampleInfo } from '../api/client';

interface Props {
  camber: number;
  camberPos: number;
  thickness: number;
  geometrySource: 'training' | 'analytical';
  matchedSample: SampleInfo | null;
  catalogCount: number;
  nacaCode: string;
  onCamberChange: (v: number) => void;
  onCamberPosChange: (v: number) => void;
  onThicknessChange: (v: number) => void;
  onGeometrySourceChange: (s: 'training' | 'analytical') => void;
  onSlidingStart?: () => void;
  onSlidingComplete?: () => void;
}

export default function AirfoilControls({
  camber, camberPos, thickness, geometrySource, matchedSample, catalogCount, nacaCode,
  onCamberChange, onCamberPosChange, onThicknessChange, onGeometrySourceChange,
  onSlidingStart, onSlidingComplete,
}: Props) {
  return (
    <View>
      <Text style={styles.sectionTitle}>Airfoil Geometry</Text>

      {/* Source toggle */}
      <View style={styles.sourceToggle}>
        <TouchableOpacity
          style={[styles.sourceBtn, geometrySource === 'training' && styles.sourceBtnActive]}
          onPress={() => onGeometrySourceChange('training')}
        >
          <Text style={[styles.sourceBtnText, geometrySource === 'training' && styles.sourceBtnTextActive]}>
            Training Data
          </Text>
          {catalogCount > 0 && (
            <View style={styles.countBadge}><Text style={styles.countBadgeText}>{catalogCount}</Text></View>
          )}
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.sourceBtn, geometrySource === 'analytical' && styles.sourceBtnActive]}
          onPress={() => onGeometrySourceChange('analytical')}
        >
          <Text style={[styles.sourceBtnText, geometrySource === 'analytical' && styles.sourceBtnTextActive]}>
            Custom NACA
          </Text>
        </TouchableOpacity>
      </View>

      {/* NACA badge */}
      <View style={styles.nacaBadge}>
        <Text style={styles.nacaBadgeLabel}>
          {geometrySource === 'training' ? 'BEST MATCH' : 'PROFILE'}
        </Text>
        <Text style={styles.nacaBadgeCode}>
          {geometrySource === 'training' && matchedSample ? `#${matchedSample.idx}` : `NACA ${nacaCode}`}
        </Text>
      </View>

      {/* Match info */}
      {geometrySource === 'training' && matchedSample && (
        <View style={styles.matchInfo}>
          {([
            ['Thickness', `${matchedSample.thickness.toFixed(1)}%`],
            ['Camber', `${matchedSample.camber.toFixed(1)}%`],
            ['Training AoA', `${matchedSample.aoa.toFixed(1)}°`],
            ['Training V', `${matchedSample.velocity.toFixed(0)} m/s`],
          ] as [string, string][]).map(([label, val]) => (
            <View key={label} style={styles.matchRow}>
              <Text style={styles.matchLabel}>{label}</Text>
              <Text style={styles.matchValue}>{val}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Camber */}
      <View style={styles.sliderRow}>
        <Text style={styles.label}>Max Camber (m)</Text>
        <Text style={styles.value}>{camber}%</Text>
      </View>
      <Slider
        style={styles.slider}
        minimumValue={0} maximumValue={6} step={1} value={camber}
        onValueChange={v => onCamberChange(Math.round(v))}
        onSlidingStart={onSlidingStart} onSlidingComplete={onSlidingComplete}
        minimumTrackTintColor={COLORS.accent} maximumTrackTintColor="#2a2d3a" thumbTintColor={COLORS.accentLight}
      />

      {/* Camber position */}
      <View style={[styles.sliderRow, camber === 0 && styles.dimmed]}>
        <Text style={styles.label}>Camber Position (p)</Text>
        <Text style={styles.value}>{camberPos * 10}% chord</Text>
      </View>
      <Slider
        style={[styles.slider, camber === 0 && styles.dimmed]}
        minimumValue={1} maximumValue={7} step={1} value={camberPos}
        onValueChange={v => onCamberPosChange(Math.round(v))}
        disabled={camber === 0}
        onSlidingStart={onSlidingStart} onSlidingComplete={onSlidingComplete}
        minimumTrackTintColor={COLORS.accent} maximumTrackTintColor="#2a2d3a" thumbTintColor={COLORS.accentLight}
      />

      {/* Thickness */}
      <View style={styles.sliderRow}>
        <Text style={styles.label}>Max Thickness (t)</Text>
        <Text style={styles.value}>{thickness}%</Text>
      </View>
      <Slider
        style={styles.slider}
        minimumValue={6} maximumValue={24} step={1} value={thickness}
        onValueChange={v => onThicknessChange(Math.round(v))}
        onSlidingStart={onSlidingStart} onSlidingComplete={onSlidingComplete}
        minimumTrackTintColor={COLORS.accent} maximumTrackTintColor="#2a2d3a" thumbTintColor={COLORS.accentLight}
      />

      {/* Presets */}
      <Text style={styles.presetsLabel}>Quick Presets</Text>
      <View style={styles.presetRow}>
        {NACA_PRESETS.map(p => (
          <TouchableOpacity
            key={p.label}
            style={[styles.presetChip, camber === p.m && thickness === p.t && styles.presetChipActive]}
            onPress={() => { onCamberChange(p.m); onCamberPosChange(p.p); onThicknessChange(p.t); }}
          >
            <Text style={[styles.presetChipText, camber === p.m && thickness === p.t && styles.presetChipTextActive]}>
              {p.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  sectionTitle: { color: COLORS.textDim, fontSize: 11, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8, paddingBottom: 5, borderBottomWidth: 1, borderBottomColor: COLORS.border, marginTop: 18 },
  sliderRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline', marginTop: 8 },
  label: { color: COLORS.textMuted, fontSize: 13, fontWeight: '500' },
  value: { color: COLORS.accentMuted, fontSize: 14, fontWeight: '700', fontFamily: FONTS.mono },
  slider: { width: '100%', height: 36 },
  dimmed: { opacity: 0.35 },
  sourceToggle: { flexDirection: 'row', gap: 6, padding: 4, borderRadius: 10, backgroundColor: COLORS.bgSecondary, borderWidth: 1, borderColor: COLORS.border },
  sourceBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, paddingVertical: 13, borderRadius: 8, backgroundColor: 'transparent' },
  sourceBtnActive: { backgroundColor: COLORS.accent },
  sourceBtnText: { color: COLORS.textDim, fontSize: 13, fontWeight: '600' },
  sourceBtnTextActive: { color: '#fff' },
  countBadge: { paddingHorizontal: 5, paddingVertical: 2, borderRadius: 99, backgroundColor: 'rgba(255,255,255,0.15)' },
  countBadgeText: { color: '#fff', fontSize: 10, fontWeight: '700', fontFamily: FONTS.mono },
  nacaBadge: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 10, marginVertical: 8, borderRadius: 8, borderWidth: 1, borderColor: COLORS.borderAccent, backgroundColor: 'rgba(99,102,241,0.06)' },
  nacaBadgeLabel: { color: COLORS.textDim, fontSize: 10, fontWeight: '700', letterSpacing: 1 },
  nacaBadgeCode: { color: COLORS.accentMuted, fontSize: 18, fontWeight: '800', fontFamily: FONTS.mono },
  matchInfo: { padding: 10, borderRadius: 8, backgroundColor: 'rgba(16,185,129,0.06)', borderWidth: 1, borderColor: 'rgba(16,185,129,0.2)', marginBottom: 8 },
  matchRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 2 },
  matchLabel: { color: COLORS.textDim, fontSize: 12 },
  matchValue: { color: COLORS.success, fontSize: 12, fontWeight: '700', fontFamily: FONTS.mono },
  presetsLabel: { color: COLORS.textDim, fontSize: 11, fontWeight: '600', marginTop: 12, marginBottom: 6 },
  presetRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  presetChip: { paddingHorizontal: 14, paddingVertical: 11, borderRadius: 99, borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)', backgroundColor: 'rgba(255,255,255,0.03)' },
  presetChipActive: { borderColor: COLORS.accent, backgroundColor: 'rgba(99,102,241,0.15)' },
  presetChipText: { color: COLORS.textDim, fontSize: 13, fontWeight: '600' },
  presetChipTextActive: { color: COLORS.accentMuted },
});

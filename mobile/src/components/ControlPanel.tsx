import React, { useState, useRef } from 'react';
import { View, Text, ScrollView, TouchableOpacity, Switch, StyleSheet } from 'react-native';
import { COLORS, FONTS } from '../constants/theme';
import { COLORMAP_OPTIONS } from '../hooks/usePrediction';
import { usePrediction } from '../hooks/usePrediction';
import BoundaryControls from './BoundaryControls';
import AirfoilControls from './AirfoilControls';
import ServerStatus from './ServerStatus';

type PredictionReturn = ReturnType<typeof usePrediction>;

interface Props {
  prediction: PredictionReturn;
}

export default function ControlPanel({ prediction }: Props) {
  const p = prediction;
  const [scrollEnabled, setScrollEnabled] = useState(true);

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.logoRow}>
          <View style={styles.logo}><Text style={styles.logoText}>A</Text></View>
          <View>
            <Text style={styles.title}>AeroML</Text>
            <Text style={styles.subtitle}>Real-Time Aerodynamic Prediction</Text>
          </View>
        </View>
      </View>

      {/* Server status */}
      <View style={styles.section}>
        <ServerStatus
          serverOnline={p.serverOnline}
          deviceName={p.deviceName}
          inferenceTime={p.inferenceTime}
          showSettings={p.showSettings}
          serverUrl={p.serverUrl}
          onToggleSettings={() => p.setShowSettings(s => !s)}
          onServerUrlChange={p.setServerUrl}
          onSave={p.saveServerUrl}
        />
      </View>

      {/* Scrollable controls */}
      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} scrollEnabled={scrollEnabled} nestedScrollEnabled>
        <BoundaryControls
          aoa={p.aoa}
          velocity={p.velocity}
          onAoaChange={p.setAoa}
          onVelocityChange={p.setVelocity}
          onSlidingStart={() => setScrollEnabled(false)}
          onSlidingComplete={() => setScrollEnabled(true)}
        />

        <AirfoilControls
          camber={p.camber}
          camberPos={p.camberPos}
          thickness={p.thickness}
          geometrySource={p.geometrySource}
          matchedSample={p.matchedSample}
          catalogCount={p.catalog.length}
          nacaCode={p.nacaCode}
          onCamberChange={p.setCamber}
          onCamberPosChange={p.setCamberPos}
          onThicknessChange={p.setThickness}
          onGeometrySourceChange={p.setGeometrySource}
          onSlidingStart={() => setScrollEnabled(false)}
          onSlidingComplete={() => setScrollEnabled(true)}
        />

        {/* Colormap */}
        <Text style={[styles.sectionTitle, { marginTop: 18 }]}>Colormap</Text>
        <View style={styles.cmapRow}>
          {COLORMAP_OPTIONS.map(cm => (
            <TouchableOpacity
              key={cm}
              style={[styles.cmapChip, p.colormaps[p.activeField] === cm && styles.cmapChipActive]}
              onPress={() => p.setColormapForField(p.activeField, cm)}
            >
              <Text style={[styles.cmapChipText, p.colormaps[p.activeField] === cm && styles.cmapChipTextActive]}>
                {cm}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Particle toggle */}
        <View style={styles.toggleRow}>
          <Text style={styles.toggleLabel}>Flow Particles</Text>
          <Switch
            value={p.showParticles}
            onValueChange={p.setShowParticles}
            trackColor={{ false: '#2a2d3a', true: 'rgba(99,102,241,0.4)' }}
            thumbColor={p.showParticles ? COLORS.accent : '#555'}
          />
        </View>

        {/* Computed inputs */}
        <View style={styles.computedRow}>
          <View style={styles.computedCard}>
            <Text style={styles.computedLabel}>Ux (m/s)</Text>
            <Text style={styles.computedValue}>{p.uxIn.toFixed(1)}</Text>
          </View>
          <View style={styles.computedCard}>
            <Text style={styles.computedLabel}>Uy (m/s)</Text>
            <Text style={styles.computedValue}>{p.uyIn.toFixed(1)}</Text>
          </View>
        </View>

        <View style={{ height: 30 }} />
      </ScrollView>

      {/* Error banner */}
      {p.error && (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>{p.error}</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 4, backgroundColor: COLORS.bg, borderRightWidth: 1, borderRightColor: COLORS.border },
  header: { paddingHorizontal: 16, paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  logoRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  logo: { width: 32, height: 32, borderRadius: 7, backgroundColor: COLORS.accent, alignItems: 'center', justifyContent: 'center' },
  logoText: { color: '#fff', fontWeight: '800', fontSize: 16 },
  title: { color: COLORS.text, fontSize: 17, fontWeight: '700' },
  subtitle: { color: COLORS.textDim, fontSize: 10, fontWeight: '500' },
  section: { paddingHorizontal: 16, paddingTop: 10 },
  scroll: { flex: 1 },
  scrollContent: { paddingHorizontal: 16 },
  computedRow: { flexDirection: 'row', gap: 10, marginTop: 16 },
  computedCard: { flex: 1, padding: 10, borderRadius: 8, borderWidth: 1, borderColor: COLORS.border, backgroundColor: COLORS.bgSecondary },
  computedLabel: { color: COLORS.textDim, fontSize: 11, fontWeight: '600', marginBottom: 3 },
  computedValue: { color: COLORS.text, fontSize: 14, fontWeight: '700', fontFamily: FONTS.mono },
  unit: { color: COLORS.textDim, fontSize: 11, fontWeight: '500' },
  errorBanner: { backgroundColor: 'rgba(239,68,68,0.1)', borderTopWidth: 1, borderTopColor: 'rgba(239,68,68,0.25)', paddingHorizontal: 16, paddingVertical: 8 },
  errorText: { color: COLORS.error, fontSize: 12, fontFamily: FONTS.mono },
  sectionTitle: { color: COLORS.textDim, fontSize: 11, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8, paddingBottom: 5, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  cmapRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  cmapChip: { paddingHorizontal: 14, paddingVertical: 11, borderRadius: 99, borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)', backgroundColor: 'rgba(255,255,255,0.03)' },
  cmapChipActive: { borderColor: COLORS.accent, backgroundColor: 'rgba(99,102,241,0.15)' },
  cmapChipText: { color: COLORS.textDim, fontSize: 13, fontWeight: '600' },
  cmapChipTextActive: { color: COLORS.accentMuted },
  toggleRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 14 },
  toggleLabel: { color: COLORS.textMuted, fontSize: 13, fontWeight: '500' },
});

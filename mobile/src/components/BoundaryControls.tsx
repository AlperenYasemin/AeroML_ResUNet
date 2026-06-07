import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Slider from '@react-native-community/slider';
import { COLORS, FONTS } from '../constants/theme';

interface Props {
  aoa: number;
  velocity: number;
  onAoaChange: (v: number) => void;
  onVelocityChange: (v: number) => void;
  onSlidingStart?: () => void;
  onSlidingComplete?: () => void;
}

export default function BoundaryControls({ aoa, velocity, onAoaChange, onVelocityChange, onSlidingStart, onSlidingComplete }: Props) {
  return (
    <View>
      <Text style={styles.sectionTitle}>Boundary Conditions</Text>

      <View style={styles.sliderRow}>
        <Text style={styles.label}>Angle of Attack (α)</Text>
        <Text style={styles.value}>{aoa.toFixed(1)}°</Text>
      </View>
      <Slider
        style={styles.slider}
        minimumValue={-5} maximumValue={15} step={0.5} value={aoa}
        onValueChange={onAoaChange}
        onSlidingStart={onSlidingStart} onSlidingComplete={onSlidingComplete}
        minimumTrackTintColor={COLORS.accent} maximumTrackTintColor="#2a2d3a" thumbTintColor={COLORS.accentLight}
      />

      <View style={styles.sliderRow}>
        <Text style={styles.label}>Freestream Velocity (U∞)</Text>
        <Text style={styles.value}>{velocity.toFixed(0)} m/s</Text>
      </View>
      <Slider
        style={styles.slider}
        minimumValue={10} maximumValue={80} step={1} value={velocity}
        onValueChange={onVelocityChange}
        onSlidingStart={onSlidingStart} onSlidingComplete={onSlidingComplete}
        minimumTrackTintColor={COLORS.accent} maximumTrackTintColor="#2a2d3a" thumbTintColor={COLORS.accentLight}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  sectionTitle: { color: COLORS.textDim, fontSize: 11, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8, paddingBottom: 5, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  sliderRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline', marginTop: 8 },
  label: { color: COLORS.textMuted, fontSize: 13, fontWeight: '500' },
  value: { color: COLORS.accentMuted, fontSize: 14, fontWeight: '700', fontFamily: FONTS.mono },
  slider: { width: '100%', height: 36 },
});

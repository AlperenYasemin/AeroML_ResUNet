import React, { useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { COLORS, FIELDS, FieldKey } from '../constants/theme';
import { RawPrediction } from '../api/client';
import FieldCanvas, { FieldCanvasRef } from './FieldCanvas';

interface Props {
  rawData: RawPrediction | null;
  activeField: FieldKey;
  colormap: string;
  showParticles: boolean;
  isLoading: boolean;
  serverOnline: boolean | null;
  onFieldChange: (field: FieldKey) => void;
  onError: (msg: string) => void;
}

export default function VisualizationPanel({
  rawData, activeField, colormap, showParticles, isLoading, serverOnline, onFieldChange, onError,
}: Props) {
  return (
    <View style={styles.container}>
      {/* Field tabs */}
      <View style={styles.fieldTabs}>
        {FIELDS.map(f => (
          <TouchableOpacity
            key={f.key}
            style={[styles.fieldTab, activeField === f.key && styles.fieldTabActive]}
            onPress={() => onFieldChange(f.key)}
          >
            <Text style={[styles.fieldTabText, activeField === f.key && styles.fieldTabTextActive]}>
              {f.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Canvas */}
      <FieldCanvas
        rawData={rawData}
        activeField={activeField}
        colormap={colormap}
        showParticles={showParticles}
        isLoading={isLoading}
        onError={onError}
      />

      {/* Placeholder when no data */}
      {!rawData && !isLoading && (
        <View style={styles.placeholder}>
          <Text style={styles.placeholderText}>
            {serverOnline === false
              ? 'Backend server is offline.'
              : 'Adjust parameters to begin.'}
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 6, backgroundColor: COLORS.bgCanvas },
  fieldTabs: { flexDirection: 'row', paddingHorizontal: 10, paddingVertical: 6, gap: 6, backgroundColor: '#14161f', borderBottomWidth: 1, borderBottomColor: COLORS.border },
  fieldTab: { flex: 1, paddingVertical: 12, borderRadius: 8, alignItems: 'center', backgroundColor: 'transparent' },
  fieldTabActive: { backgroundColor: COLORS.accent },
  fieldTabText: { color: COLORS.textDim, fontSize: 13, fontWeight: '600' },
  fieldTabTextActive: { color: '#fff' },
  placeholder: { ...StyleSheet.absoluteFillObject, top: 40, alignItems: 'center', justifyContent: 'center' },
  placeholderText: { color: COLORS.textDim, fontSize: 13 },
});

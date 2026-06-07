import React from 'react';
import { View, Text, TextInput, TouchableOpacity, Modal, StyleSheet } from 'react-native';
import { COLORS, FONTS } from '../constants/theme';

interface Props {
  serverOnline: boolean | null;
  deviceName: string;
  inferenceTime: number | null;
  showSettings: boolean;
  serverUrl: string;
  onToggleSettings: () => void;
  onServerUrlChange: (url: string) => void;
  onSave: () => void;
}

export default function ServerStatus({
  serverOnline, deviceName, inferenceTime, showSettings, serverUrl,
  onToggleSettings, onServerUrlChange, onSave,
}: Props) {
  return (
    <>
      <View style={styles.statusRow}>
        {inferenceTime !== null && (
          <View style={styles.badge}>
            <View style={[styles.dot, { backgroundColor: COLORS.success }]} />
            <Text style={styles.badgeText}>{inferenceTime.toFixed(1)} ms</Text>
          </View>
        )}
        <TouchableOpacity onPress={onToggleSettings}>
          <View style={[styles.badge, {
            borderColor: serverOnline ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.4)',
          }]}>
            <View style={[styles.dot, { backgroundColor: serverOnline ? COLORS.success : COLORS.error }]} />
            <Text style={[styles.badgeText, { color: serverOnline ? COLORS.success : COLORS.error }]}>
              {serverOnline ? deviceName?.toUpperCase() || 'Online' : 'Offline'}
            </Text>
          </View>
        </TouchableOpacity>
      </View>

      <Modal visible={showSettings} transparent animationType="fade" onRequestClose={onToggleSettings}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Server Configuration</Text>
            <TextInput
              style={styles.textInput}
              value={serverUrl}
              onChangeText={onServerUrlChange}
              autoCapitalize="none"
              autoCorrect={false}
              placeholder="http://192.168.1.x:8000"
              placeholderTextColor="#555"
            />
            <View style={styles.modalButtons}>
              <TouchableOpacity style={styles.cancelBtn} onPress={onToggleSettings}>
                <Text style={styles.cancelBtnText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.saveBtn} onPress={onSave}>
                <Text style={styles.saveBtnText}>Connect</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  statusRow: { flexDirection: 'row', gap: 8, marginBottom: 10, flexWrap: 'wrap' },
  badge: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 12, paddingVertical: 9, borderRadius: 99, borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)', backgroundColor: 'rgba(255,255,255,0.03)' },
  badgeText: { color: COLORS.textMuted, fontSize: 12, fontFamily: FONTS.mono, fontWeight: '600' },
  dot: { width: 6, height: 6, borderRadius: 3 },
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'center', alignItems: 'center' },
  modalContent: { backgroundColor: COLORS.bgSecondary, borderRadius: 12, padding: 20, width: 340, borderWidth: 1, borderColor: COLORS.border },
  modalTitle: { color: COLORS.text, fontSize: 16, fontWeight: '700', marginBottom: 12 },
  textInput: { backgroundColor: COLORS.bg, borderWidth: 1, borderColor: 'rgba(255,255,255,0.12)', borderRadius: 8, padding: 10, color: COLORS.text, fontSize: 13, fontFamily: FONTS.mono, marginBottom: 12 },
  modalButtons: { flexDirection: 'row', gap: 8 },
  cancelBtn: { flex: 1, padding: 10, borderRadius: 8, alignItems: 'center', borderWidth: 1, borderColor: COLORS.border },
  cancelBtnText: { color: COLORS.textMuted, fontWeight: '600', fontSize: 13 },
  saveBtn: { flex: 1, backgroundColor: COLORS.accent, borderRadius: 8, padding: 10, alignItems: 'center' },
  saveBtnText: { color: '#fff', fontWeight: '600', fontSize: 13 },
});

import React, { useRef, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react';
import { View, Text, StyleSheet, Platform, ActivityIndicator } from 'react-native';
import { COLORS, FONTS, FIELDS, FieldKey } from '../constants/theme';
import { RawPrediction } from '../api/client';
import { RENDERER_HTML } from '../utils/rendererHtml';

let WebView: any;
if (Platform.OS !== 'web') {
  WebView = require('react-native-webview').default;
}

interface Props {
  rawData: RawPrediction | null;
  activeField: FieldKey;
  colormap: string;
  showParticles: boolean;
  isLoading: boolean;
  onError: (msg: string) => void;
}

export interface FieldCanvasRef {
  sendToWebView: (data: RawPrediction) => void;
}

const FieldCanvas = forwardRef<FieldCanvasRef, Props>(({ rawData, activeField, colormap, showParticles, isLoading, onError }, ref) => {
  const webViewRef = useRef<any>(null);
  const activeFieldRef = useRef<FieldKey>(activeField);
  const colormapRef = useRef(colormap);
  const rawDataRef = useRef(rawData);
  useEffect(() => { activeFieldRef.current = activeField; }, [activeField]);
  useEffect(() => { colormapRef.current = colormap; }, [colormap]);
  useEffect(() => { rawDataRef.current = rawData; }, [rawData]);

  const sendToWebView = useCallback((data: RawPrediction) => {
    if (!webViewRef.current || !data) return;
    const field = activeFieldRef.current;
    const fieldMap: Record<FieldKey, { field: string; min: number; max: number }> = {
      pressure: { field: data.pressure, min: data.pressure_min, max: data.pressure_max },
      velocity_x: { field: data.velocity_x, min: data.velocity_x_min, max: data.velocity_x_max },
      velocity_y: { field: data.velocity_y, min: data.velocity_y_min, max: data.velocity_y_max },
    };
    const f = fieldMap[field];
    const cmap = colormapRef.current;
    const msg = JSON.stringify({
      type: 'render', field: f.field, sdf: data.sdf, vmin: f.min, vmax: f.max, cmap,
      vx: data.velocity_x, vy: data.velocity_y,
    });

    if (Platform.OS === 'web') {
      const iframe = webViewRef.current as any;
      if (iframe?.contentWindow) iframe.contentWindow.postMessage(msg, '*');
    } else {
      (webViewRef.current as any).postMessage(msg);
    }
  }, []);

  useImperativeHandle(ref, () => ({ sendToWebView }), [sendToWebView]);

  useEffect(() => {
    if (!rawData) return;
    sendToWebView(rawData);
    if (Platform.OS === 'web') {
      const t1 = setTimeout(() => sendToWebView(rawData), 300);
      const t2 = setTimeout(() => sendToWebView(rawData), 800);
      return () => { clearTimeout(t1); clearTimeout(t2); };
    }
  }, [activeField, colormap, rawData, sendToWebView]);

  // Send particle toggle config to WebView
  useEffect(() => {
    if (!webViewRef.current) return;
    const msg = JSON.stringify({ type: 'config', showParticles });
    if (Platform.OS === 'web') {
      const iframe = webViewRef.current as any;
      if (iframe?.contentWindow) iframe.contentWindow.postMessage(msg, '*');
    } else {
      (webViewRef.current as any).postMessage(msg);
    }
  }, [showParticles]);

  // Web iframe message handler — also detects 'ready' to flush pending data
  useEffect(() => {
    if (Platform.OS !== 'web') return;
    const handler = (e: any) => {
      if (!e.data) return;
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'error') onError(`Render error: ${msg.msg}`);
      } catch {}
    };
    (window as any).addEventListener('message', handler);
    return () => (window as any).removeEventListener('message', handler);
  }, [onError, sendToWebView, showParticles]);

  return (
    <View style={styles.container}>
      {isLoading && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color={COLORS.accent} />
        </View>
      )}

      {Platform.OS === 'web' ? (
        <iframe
          ref={webViewRef}
          srcDoc={RENDERER_HTML}
style={{ flex: 1, border: 'none', width: '100%', height: '100%', backgroundColor: COLORS.bgCanvas } as any}
        />
      ) : WebView ? (
        <WebView
          ref={webViewRef}
          source={{ html: RENDERER_HTML }}
          style={styles.webview}
          scrollEnabled={false}
          bounces={false}
          javaScriptEnabled={true}
          originWhitelist={['*']}
          onMessage={(event: any) => {
            try {
              const msg = JSON.parse(event.nativeEvent.data);
              if (msg.type === 'error') onError(`Render error: ${msg.msg}`);
              // Flush latest data once the WebView renderer signals it's ready —
              // avoids a dropped first render if prediction arrives before load.
              else if (msg.type === 'ready' && rawDataRef.current) sendToWebView(rawDataRef.current);
            } catch {}
          }}
        />
      ) : null}

      {/* Colorbar */}
      {rawData && (
        <View style={styles.colorbar}>
          <Text style={styles.colorbarText}>
            {activeField === 'pressure' ? rawData.pressure_min.toFixed(3) :
             activeField === 'velocity_x' ? rawData.velocity_x_min.toFixed(3) :
             rawData.velocity_y_min.toFixed(3)}
          </Text>
          <View style={styles.colorbarGradient} />
          <Text style={styles.colorbarText}>
            {activeField === 'pressure' ? rawData.pressure_max.toFixed(3) :
             activeField === 'velocity_x' ? rawData.velocity_x_max.toFixed(3) :
             rawData.velocity_y_max.toFixed(3)}
          </Text>
        </View>
      )}
    </View>
  );
});

export default FieldCanvas;

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bgCanvas },
  webview: { flex: 1, backgroundColor: 'transparent' },
  loadingOverlay: { ...StyleSheet.absoluteFillObject, zIndex: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(15,17,23,0.7)' },
  colorbar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 10, paddingVertical: 4, backgroundColor: COLORS.bgCanvas },
  colorbarText: { color: COLORS.textDim, fontSize: 9, fontFamily: FONTS.mono },
  colorbarGradient: { flex: 1, height: 3, marginHorizontal: 6, borderRadius: 2, backgroundColor: '#2a2d3a' },
});

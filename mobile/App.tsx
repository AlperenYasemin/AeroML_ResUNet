import React from 'react';
import { SafeAreaView, StatusBar, StyleSheet } from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { usePrediction } from './src/hooks/usePrediction';
import ControlPanel from './src/components/ControlPanel';
import VisualizationPanel from './src/components/VisualizationPanel';
import { COLORS } from './src/constants/theme';

export default function App() {
  const prediction = usePrediction();

  return (
    <GestureHandlerRootView style={styles.container}>
    <SafeAreaView style={styles.inner}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} hidden />

      {/* Left panel — 30% controls */}
      <ControlPanel prediction={prediction} />

      {/* Right panel — 70% visualization */}
      <VisualizationPanel
        rawData={prediction.rawData}
        activeField={prediction.activeField}
        colormap={prediction.colormaps[prediction.activeField]}
        showParticles={prediction.showParticles}
        isLoading={prediction.isLoading}
        serverOnline={prediction.serverOnline}
        onFieldChange={prediction.setActiveField}
        onError={prediction.setError}
      />
    </SafeAreaView>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.bg,
  },
  inner: {
    flex: 1,
    flexDirection: 'row',
  },
});

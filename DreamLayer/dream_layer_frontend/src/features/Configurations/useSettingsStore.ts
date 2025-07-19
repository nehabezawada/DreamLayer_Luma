
import { create } from 'zustand';

interface SettingsState {
  modelPath: string;
  enableLowVRAM: boolean;
  uiTheme: string;
  language: string;
  showQuickSettings: boolean;
  showProgressInTitle: boolean;
  computeDevice: string;
  vramUsageTarget: number;
  parallelProcessing: number;
  useXformers: boolean;
  optimizeMedVram: boolean;
  outputDirectory: string;
  modelsDirectory: string;
  controlNetModelsPath: string;
  upscalerModelsPath: string;
  vaeModelsPath: string;
  loraEmbeddingsPath: string;
  filenameFormat: string;
  saveMetadata: boolean;
  updateChannel: string;
  autoUpdate: boolean;
  
  // Setters
  setModelPath: (path: string) => void;
  setEnableLowVRAM: (enabled: boolean) => void;
  setUiTheme: (theme: string) => void;
  setLanguage: (lang: string) => void;
  setShowQuickSettings: (show: boolean) => void;
  setShowProgressInTitle: (show: boolean) => void;
  setComputeDevice: (device: string) => void;
  setVramUsageTarget: (target: number) => void;
  setParallelProcessing: (count: number) => void;
  setUseXformers: (use: boolean) => void;
  setOptimizeMedVram: (optimize: boolean) => void;
  setOutputDirectory: (dir: string) => void;
  setModelsDirectory: (dir: string) => void;
  setControlNetModelsPath: (path: string) => void;
  setUpscalerModelsPath: (path: string) => void;
  setVaeModelsPath: (path: string) => void;
  setLoraEmbeddingsPath: (path: string) => void;
  setFilenameFormat: (format: string) => void;
  setSaveMetadata: (save: boolean) => void;
  setUpdateChannel: (channel: string) => void;
  setAutoUpdate: (auto: boolean) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  // Initial state
  modelPath: 'flux1-dev-fp8.safetensors',
  enableLowVRAM: false,
  uiTheme: 'system',
  language: 'en',
  showQuickSettings: true,
  showProgressInTitle: true,
  computeDevice: 'cuda',
  vramUsageTarget: 85,
  parallelProcessing: 2,
  useXformers: true,
  optimizeMedVram: false,
  outputDirectory: '/path/to/outputs',
  modelsDirectory: '/path/to/parent/of/models',
  controlNetModelsPath: '/path/to/models/ControlNet',
  upscalerModelsPath: '/path/to/models/ESRGAN',
  vaeModelsPath: '/path/to/models/VAE',
  loraEmbeddingsPath: '/path/to/models/Lora',
  filenameFormat: 'paste the path if it is outside project root',
  saveMetadata: true,
  updateChannel: 'stable',
  autoUpdate: false,

  // Setters
  setModelPath: (path) => set({ modelPath: path }),
  setEnableLowVRAM: (enabled) => set({ enableLowVRAM: enabled }),
  setUiTheme: (theme) => set({ uiTheme: theme }),
  setLanguage: (lang) => set({ language: lang }),
  setShowQuickSettings: (show) => set({ showQuickSettings: show }),
  setShowProgressInTitle: (show) => set({ showProgressInTitle: show }),
  setComputeDevice: (device) => set({ computeDevice: device }),
  setVramUsageTarget: (target) => set({ vramUsageTarget: target }),
  setParallelProcessing: (count) => set({ parallelProcessing: count }),
  setUseXformers: (use) => set({ useXformers: use }),
  setOptimizeMedVram: (optimize) => set({ optimizeMedVram: optimize }),
  setOutputDirectory: (dir) => set({ outputDirectory: dir }),
  setModelsDirectory: (dir) => set({ modelsDirectory: dir }),
  setControlNetModelsPath: (path) => set({ controlNetModelsPath: path }),
  setUpscalerModelsPath: (path) => set({ upscalerModelsPath: path }),
  setVaeModelsPath: (path) => set({ vaeModelsPath: path }),
  setLoraEmbeddingsPath: (path) => set({ loraEmbeddingsPath: path }),
  setFilenameFormat: (format) => set({ filenameFormat: format }),
  setSaveMetadata: (save) => set({ saveMetadata: save }),
  setUpdateChannel: (channel) => set({ updateChannel: channel }),
  setAutoUpdate: (auto) => set({ autoUpdate: auto }),
}));

import { create } from 'zustand';
import { LoraRequest } from '@/types/lora';

interface LoraState {
  loraConfig: LoraRequest | null;
  setLoraConfig: (config: LoraRequest | null) => void;
  setSelectedLora: (filename: string | null, defaultWeight?: number) => void;
}

const useLoraStore = create<LoraState>((set) => ({
  loraConfig: null,
  setLoraConfig: (config) => set({ loraConfig: config }),
  setSelectedLora: (filename, defaultWeight = 1.0) => set((state) => {
    if (!filename) {
      return { loraConfig: null };
    }
    
    return {
      loraConfig: {
        enabled: true,
        lora_name: filename,
        strength_model: defaultWeight,
        strength_clip: defaultWeight
      }
    };
  }),
}));

export default useLoraStore; 
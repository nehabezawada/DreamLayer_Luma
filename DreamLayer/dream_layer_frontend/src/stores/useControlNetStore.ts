import { create } from 'zustand';
import { ControlNetUnit, ControlNetRequest, createControlNetUnit } from '@/types/controlnet';

interface ControlNetState {
  // Main ControlNet configuration
  activeUnits: number[];
  setActiveUnits: (units: number[]) => void;
  
  // Selected unit for the UI
  selectedUnitIndex: number;
  setSelectedUnitIndex: (index: number) => void;
  
  // All ControlNet units configuration
  controlNetUnits: Record<number, ControlNetUnit>;
  setControlNetUnits: (units: Record<number, ControlNetUnit>) => void;
  updateControlNetUnit: (unitIndex: number, updatedUnit: Partial<ControlNetUnit>) => void;
  
  // ControlNet request configuration (for API)
  controlNetConfig: ControlNetRequest | null;
  setControlNetConfig: (config: ControlNetRequest | null) => void;
  
  // Extension state
  isControlNetEnabled: boolean;
  setIsControlNetEnabled: (enabled: boolean) => void;
  isControlNetOpen: boolean;
  setIsControlNetOpen: (open: boolean) => void;
  
  // Helper actions
  addUnit: () => void;
  removeUnit: (unitIndex: number) => void;
  resetControlNet: () => void;
}

const useControlNetStore = create<ControlNetState>((set, get) => ({
  // Initial state
  activeUnits: [0],
  selectedUnitIndex: 0,
  controlNetUnits: {
    0: createControlNetUnit(0)
  },
  controlNetConfig: null,
  isControlNetEnabled: false,
  isControlNetOpen: false,
  
  // Setters
  setActiveUnits: (units) => set({ activeUnits: units }),
  setSelectedUnitIndex: (index) => set({ selectedUnitIndex: index }),
  setControlNetUnits: (units) => set({ controlNetUnits: units }),
  setControlNetConfig: (config) => set({ controlNetConfig: config }),
  setIsControlNetEnabled: (enabled) => set({ isControlNetEnabled: enabled }),
  setIsControlNetOpen: (open) => set({ isControlNetOpen: open }),
  
  // Update individual unit
  updateControlNetUnit: (unitIndex, updatedUnit) => set((state) => {
    const newState = {
      controlNetUnits: {
        ...state.controlNetUnits,
        [unitIndex]: {
          ...state.controlNetUnits[unitIndex],
          ...updatedUnit
        }
      }
    };
    
    console.log('[ControlNetStore] New state for unit:', newState.controlNetUnits[unitIndex]);
    return newState;
  }),
  
  // Add new unit
  addUnit: () => set((state) => {
    const maxUnits = 10;
    if (state.activeUnits.length >= maxUnits) return state;
    
    // Find next available unit number
    const existingUnits = new Set(state.activeUnits);
    let nextUnitNumber = 0;
    while (existingUnits.has(nextUnitNumber)) {
      nextUnitNumber++;
    }
    
    const newUnits = [...state.activeUnits, nextUnitNumber].sort((a, b) => a - b);
    
    return {
      activeUnits: newUnits,
      selectedUnitIndex: nextUnitNumber,
      controlNetUnits: {
        ...state.controlNetUnits,
        [nextUnitNumber]: createControlNetUnit(nextUnitNumber)
      }
    };
  }),
  
  // Remove unit
  removeUnit: (unitIndex) => set((state) => {
    if (state.activeUnits.length <= 1 || unitIndex === 0) return state;
    
    const newUnits = state.activeUnits.filter(idx => idx !== unitIndex);
    const newControlNetUnits = { ...state.controlNetUnits };
    delete newControlNetUnits[unitIndex];
    
    // If removed unit was selected, select first available
    const newSelectedIndex = state.selectedUnitIndex === unitIndex 
      ? newUnits[0] 
      : state.selectedUnitIndex;
    
    return {
      activeUnits: newUnits,
      selectedUnitIndex: newSelectedIndex,
      controlNetUnits: newControlNetUnits
    };
  }),
  
  // Reset all ControlNet state
  resetControlNet: () => set({
    activeUnits: [0],
    selectedUnitIndex: 0,
    controlNetUnits: {
      0: createControlNetUnit(0)
    },
    controlNetConfig: null,
    isControlNetEnabled: false,
    isControlNetOpen: false
  })
}));

export default useControlNetStore; 
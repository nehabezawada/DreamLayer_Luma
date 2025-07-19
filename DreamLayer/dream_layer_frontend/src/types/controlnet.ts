// Types for ControlNet configuration

export interface ControlNetRequest {
  enabled: boolean;
  units: ControlNetUnit[];
}

export interface ControlNetUnit {
  // Unit identification
  unit_index: number;
  enabled: boolean;
  
  // Input configuration
  input_image?: string | null; // Base64 encoded image data
  input_mode: "single" | "batch";
  batch_directory?: string;
  
  // Control configuration  
  control_type: string; // "openpose", "canny", "depth", etc.
  preprocessor: string; // "dw_openpose_full", "canny", "none", etc.
  model: string; // ControlNet model path
  
  // Processing parameters
  weight: number; // 0.0 - 2.0, default 1.0
  guidance_start: number; // 0.0 - 1.0, default 0.0  
  guidance_end: number; // 0.0 - 1.0, default 1.0
  resolution: number; // Processing resolution, default 512
  
  // Control modes
  control_mode: string; // "balanced", "prompt", "control"
  resize_mode: string; // "just_resize", "crop_resize", "resize_fill"
  
  // Preprocessor thresholds (for applicable preprocessors)
  threshold_a?: number; // Default 100
  threshold_b?: number; // Default 200
  
  // Advanced options
  pixel_perfect: boolean;
  low_vram: boolean;
  allow_preview: boolean;
  effective_region_mask: boolean;
  upload_independent_control: boolean;
  loopback: boolean;
}

// Constants for ControlNet options
export const DEFAULT_CONTROLNET_UNIT: Omit<ControlNetUnit, 'unit_index'> = {
  enabled: true,
  input_mode: "single",
  input_image: null,
  control_type: "openpose",
  preprocessor: "dw_openpose_full",
  model: "diffusion_pytorch_model.safetensors",
  weight: 1.0,
  guidance_start: 0.0,
  guidance_end: 1.0,
  resolution: 512,
  control_mode: "balanced",
  resize_mode: "crop_resize",
  threshold_a: 100,
  threshold_b: 200,
  pixel_perfect: false,
  low_vram: false,
  allow_preview: true,
  effective_region_mask: false,
  upload_independent_control: false,
  loopback: false
};

// Utility function to create a new ControlNet unit
export function createControlNetUnit(unit_index: number): ControlNetUnit {
  return {
    ...DEFAULT_CONTROLNET_UNIT,
    unit_index
  };
} 
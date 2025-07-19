export interface ExtrasRequest {
  // Input image
  image: File;
  
  // Upscaling parameters
  upscaler_model: string;    // "UpScaler 2" or "GFPGAN"
  upscaler_visibility: number;  // 0-1, controls first upscaler strength
  
  // Second upscaler parameters
  upscaler_model_2: string;    // "UpScaler 2" or "GFPGAN"
  upscaler_visibility_2: number;  // 0-1, controls second upscaler strength
  
  // Face restoration parameters
  gfpgan_visibility: number;  // 0-1, controls GFPGAN strength
  codeformer_visibility: number;  // 0-1, controls CodeFormer visibility
  codeformer_weight: number;  // 0-1, controls CodeFormer weight
  
  // Output format
  output_format: string;     // "PNG" or "JPEG"
}

// Response interface for the extras API
export interface ExtrasResponse {
  status: 'success' | 'error';
  message: string;
  data?: {
    output_image: string;      // URL to the processed image
    processing_time: number;   // Time in milliseconds
    original_size: {
      width: number;
      height: number;
    };
    new_size: {
      width: number;
      height: number;
    };
  };
  error?: {
    code: string;
    details: string;
  };
} 
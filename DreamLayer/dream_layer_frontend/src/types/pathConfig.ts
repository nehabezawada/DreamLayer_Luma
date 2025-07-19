export interface PathConfig {
  // Path Settings
  outputDirectory: string;          // Output Directory
  modelsDirectory: string;          // Models Directory
  controlNetModelsPath: string;     // ControlNet Models Path
  upscalerModelsPath: string;       // Upscaler Models Path
  vaeModelsPath: string;            // VAE Models Path
  loraEmbeddingsPath: string;       // LoRA / Embeddings Path
  filenameFormat: string;           // Filename Format
  saveMetadata: boolean;            // Save metadata with generations
}

// Default configuration
export const defaultPathConfig: PathConfig = {
  outputDirectory: "/path/to/outputs",
  modelsDirectory: "/path/to/models",
  controlNetModelsPath: "/path/to/models/ControlNet",
  upscalerModelsPath: "/path/to/models/ESRGAN",
  vaeModelsPath: "/path/to/models/VAE",
  loraEmbeddingsPath: "/path/to/models/Lora",
  filenameFormat: "[model_name]_[seed]_[prompt_words]",
  saveMetadata: true
}; 

// Define upscaling model data with descriptions and optimal settings
export interface UpscalingModelData {
  id: string;
  name: string;
  description: string;
  speed: string;
  optimalHiresSteps: string;
  optimalDenoisingStrength: string;
}

const upscalingModels: Record<string, UpscalingModelData> = {
  "4x-ultrasharp": {
    id: "4x-ultrasharp",
    name: "4x-UltraSharp",
    description: "Delivers extreme sharpness, great for textures and fine detail, but may exaggerate edges.",
    speed: "Medium",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.4–0.6"
  },
  "latent": {
    id: "latent",
    name: "Latent",
    description: "Fast built-in upscaler that works in latent space, good for quick results and general use. It's not ideal for fine detail or clean edges.",
    speed: "Fast",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.3–0.5"
  },
  "latent-antialiased": {
    id: "latent-antialiased",
    name: "Latent (antialiased)",
    description: "Adds smoothing to reduce jagged edges, great for line art and crisp edges. Less suited for highly detailed realism.",
    speed: "Fast",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.3–0.5"
  },
  "latent-bicubic": {
    id: "latent-bicubic",
    name: "Latent (bicubic)",
    description: "Offers a balance between sharpness and smoothness using bicubic interpolation. Not ideal for very high detail.",
    speed: "Fast",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.3–0.5"
  },
  "latent-bicubic-antialiased": {
    id: "latent-bicubic-antialiased",
    name: "Latent (bicubic antialiased)",
    description: "Combines smoothness and soft edge handling, useful for illustrations. Can blur textures slightly.",
    speed: "Fast",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.3–0.5"
  },
  "latent-nearest": {
    id: "latent-nearest",
    name: "Latent (nearest)",
    description: "Basic nearest-neighbor upscaler, fast and simple but produces blocky results. Good for retro-style images.",
    speed: "Very Fast",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.3–0.5"
  },
  "latent-nearest-exact": {
    id: "latent-nearest-exact",
    name: "Latent (nearest-exact)",
    description: "Sharpens structure slightly more than regular nearest. Good for debugging or stylized low-res effects.",
    speed: "Very Fast",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.3–0.5"
  },
  "none": {
    id: "none",
    name: "None",
    description: "Skips upscaling entirely. Only use this when you want to preserve original resolution or handle resizing elsewhere.",
    speed: "N/A",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.3–0.5"
  },
  "lanczos": {
    id: "lanczos",
    name: "Lanczos",
    description: "Classical resampling method, good for mathematical sharpness, but prone to ringing artifacts.",
    speed: "Moderate",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.3–0.5"
  },
  "nearest": {
    id: "nearest",
    name: "Nearest",
    description: "Basic pixel-doubling method, fast and sharp but often very blocky. Great for pixel art, bad for realism.",
    speed: "Very Fast",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.3–0.5"
  },
  "esrgan-4x": {
    id: "esrgan-4x",
    name: "ESRGAN_4x",
    description: "Enhances detail and sharpness, best for photorealistic content. May introduce artifacts in soft or stylized renders.",
    speed: "Medium",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.4–0.6"
  },
  "r-esrgan-4x-plus": {
    id: "r-esrgan-4x-plus",
    name: "R-ESRGAN 4x+",
    description: "Improves upon ESRGAN with better detail recovery. Great for realistic upscaling, but can exaggerate noise.",
    speed: "Medium",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.4–0.6"
  },
  "r-esrgan-4x-plus-anime6b": {
    id: "r-esrgan-4x-plus-anime6b",
    name: "R-ESRGAN 4x+ Anime6B",
    description: "Specialized for anime-style content, clean lines and color balance. Bad for realism or noisy inputs.",
    speed: "Medium",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.4–0.6"
  },
  "scunet-gan": {
    id: "scunet-gan",
    name: "ScuNET GAN",
    description: "Lightweight upscaler good for basic face and detail enhancement. Lower quality than ESRGAN.",
    speed: "Fast",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.3–0.5"
  },
  "scunet-psnr": {
    id: "scunet-psnr",
    name: "ScuNET PSNR",
    description: "Tuned for low artifact output with good denoising. Less detail enhancement but good for smooth images.",
    speed: "Fast",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.3–0.5"
  },
  "swinir-4x": {
    id: "swinir-4x",
    name: "SwinIR 4x",
    description: "Uses Swin Transformers for strong structural detail, excellent on clean renders. Slower and heavier.",
    speed: "Slow",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.4–0.6"
  },
  "ldsr": {
    id: "ldsr",
    name: "LDSR",
    description: "Uses latent diffusion to upscale with realism and fine texture. Great for photo accuracy, but significantly slower.",
    speed: "Slow",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.4–0.6"
  },
  "4x-ultramix-balanced": {
    id: "4x-ultramix-balanced",
    name: "4x-UltraMix Balanced",
    description: "Balances sharpness and smoothness, works well on a wide range of subjects. Not best for ultra-fine textures.",
    speed: "Medium",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.4–0.6"
  },
  "4x-ultramix-restore": {
    id: "4x-ultramix-restore",
    name: "4x-UltraMix Restore",
    description: "Ideal for restoring degraded or blurry images. May lose some sharpness.",
    speed: "Medium",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.4–0.6"
  },
  "4x-ultramix-smooth": {
    id: "4x-ultramix-smooth",
    name: "4x-UltraMix Smooth",
    description: "Aims for soft textures and clean gradients, ideal for portraits and skin. Not suitable for sharp lines.",
    speed: "Medium",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.4–0.6"
  },
  "remacri": {
    id: "remacri",
    name: "Remacri",
    description: "Optimized for landscapes and detail recovery in scenic shots. Slightly soft on fine edges.",
    speed: "Medium",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.4–0.6"
  },
  "lollypop": {
    id: "lollypop",
    name: "Lollypop",
    description: "Enhances facial features with balanced tone, good for portraits. Not suitable for sharp industrial textures.",
    speed: "Medium",
    optimalHiresSteps: "10–15",
    optimalDenoisingStrength: "0.4–0.6"
  }
};

export default upscalingModels;

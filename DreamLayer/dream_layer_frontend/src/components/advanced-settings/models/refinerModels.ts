
// Define refiner model data
export interface RefinerModelData {
  id: string;
  name: string;
  description: string;
  speed: string;
  optimalSwitchMin: number;
  optimalSwitchMax: number;
}

const refinerModels: Record<string, RefinerModelData> = {
  "none": {
    id: "none",
    name: "None",
    description: "No refinement will be applied to the generated image.",
    speed: "N/A",
    optimalSwitchMin: 0.8,
    optimalSwitchMax: 0.8
  },
  "sdxl-1.0": {
    id: "sdxl-1.0",
    name: "SDXL Refiner 1.0",
    description: "Great at adding final details to images, like textures and sharper features. It's best used after the base model has done most of the work. It's not good for full generations or early-stage prompts, and it slows things down a bit.",
    speed: "Slow",
    optimalSwitchMin: 0.5,
    optimalSwitchMax: 0.8
  },
  "sdxl-0.9": {
    id: "sdxl-0.9",
    name: "SDXL Refiner 0.9",
    description: "Improves image quality at the end of generation but isn't as sharp or clean as version 1.0. It's good for basic final polish but falls behind in fine detail.",
    speed: "Medium",
    optimalSwitchMin: 0.5,
    optimalSwitchMax: 0.8
  },
  "flux": {
    id: "flux",
    name: "FLUX Refiner",
    description: "Works faster than the others and is good for refining on weaker GPUs. It gives clean results quickly but isn't as high-quality as SDXL refiners. Best for speed and decent output.",
    speed: "Fast",
    optimalSwitchMin: 0.5,
    optimalSwitchMax: 0.8
  },
  "sdxl-turbo": {
    id: "sdxl-turbo",
    name: "SDXL Turbo Refiner",
    description: "Lightweight version of SDXL refiner - runs direct, much faster",
    speed: "Very Fast",
    optimalSwitchMin: 0.7,
    optimalSwitchMax: 0.9
  }
};

export default refinerModels;

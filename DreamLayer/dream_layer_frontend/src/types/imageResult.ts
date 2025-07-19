
export interface ImageResult {
  id: string;
  url: string;
  prompt: string;
  negativePrompt?: string;
  timestamp: number;
  settings?: Record<string, any>;
}

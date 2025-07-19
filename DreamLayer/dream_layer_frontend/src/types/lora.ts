export interface LoraModel {
  id: string;
  name: string;
  filename: string;
  preview?: string;
  triggerWords?: string[];
  description?: string;
  tags?: string[];
  defaultWeight?: number;
}

// LoRA request structure for API - matches backend expectations
export interface LoraRequest {
  enabled: boolean;
  lora_name: string;
  strength_model: number;
  strength_clip: number;
} 
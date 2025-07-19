
import React from 'react';
import Accordion from '@/components/Accordion';
import Slider from '@/components/Slider';

const CheckpointMergerTab = () => {
  return (
    <div className="mb-4">
      <div className="flex flex-col">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-base font-medium">Checkpoint Model Merger</h3>
          <button className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90">
            Merge Models
          </button>
        </div>
        
        <Accordion title="Primary Models" number="1" defaultOpen={true}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label htmlFor="model1" className="mb-1 block text-sm font-medium">Model A:</label>
              <select id="model1" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="">Select a checkpoint...</option>
                <option value="sd15">StableDiffusion v1.5</option>
                <option value="sd21">StableDiffusion v2.1</option>
                <option value="sdxl">StableDiffusion XL</option>
              </select>
            </div>
            
            <div>
              <label htmlFor="model2" className="mb-1 block text-sm font-medium">Model B:</label>
              <select id="model2" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="">Select a checkpoint...</option>
                <option value="sd15">StableDiffusion v1.5</option>
                <option value="sd21">StableDiffusion v2.1</option>
                <option value="sdxl">StableDiffusion XL</option>
              </select>
            </div>
          </div>
          
          <div className="mb-4">
            <Slider
              min={0}
              max={1}
              defaultValue={0.5}
              label="Interpolation (A → B)"
              sublabel="| 0 = 100% Model A, 1 = 100% Model B"
            />
          </div>
        </Accordion>
        
        <Accordion title="Tertiary Model (Optional)" number="2">
          <div className="mb-4">
            <label htmlFor="model3" className="mb-1 block text-sm font-medium">Model C:</label>
            <select id="model3" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
              <option value="">None (Optional)</option>
              <option value="sd15">StableDiffusion v1.5</option>
              <option value="sd21">StableDiffusion v2.1</option>
              <option value="sdxl">StableDiffusion XL</option>
            </select>
          </div>
          
          <div className="mb-4">
            <Slider
              min={0}
              max={1}
              defaultValue={0.25}
              label="Model C Influence"
            />
          </div>
        </Accordion>
        
        <Accordion title="Merge Method" number="3" defaultOpen={true}>
          <div className="mb-4">
            <label htmlFor="mergeMethod" className="mb-1 block text-sm font-medium">Merge Algorithm:</label>
            <select id="mergeMethod" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
              <option value="weighted">Weighted Sum</option>
              <option value="add">Add Difference</option>
              <option value="tensor">Tensor Combination</option>
            </select>
          </div>
          
          <div className="mb-4">
            <label htmlFor="customScript" className="mb-1 block text-sm font-medium">Custom Merge Script (Optional):</label>
            <textarea
              id="customScript"
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm min-h-[100px]"
              placeholder="# Python merge script (advanced)..."
            ></textarea>
          </div>
        </Accordion>
        
        <Accordion title="Output Settings" number="4" defaultOpen={true}>
          <div className="mb-4">
            <label htmlFor="outputName" className="mb-1 block text-sm font-medium">Output Model Name:</label>
            <input
              id="outputName"
              type="text"
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              placeholder="merged-model-v1"
            />
          </div>
          
          <div className="mb-4">
            <label htmlFor="saveFormat" className="mb-1 block text-sm font-medium">Save Format:</label>
            <select id="saveFormat" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
              <option value="safetensors">safetensors (Recommended)</option>
              <option value="ckpt">ckpt</option>
              <option value="pt">pt</option>
            </select>
          </div>
          
          <div className="flex items-center mb-4">
            <input type="checkbox" id="fp16" className="mr-2" checked />
            <label htmlFor="fp16" className="text-sm">Save in FP16 (half) precision</label>
          </div>
          
          <div className="flex items-center">
            <input type="checkbox" id="pruneEMA" className="mr-2" />
            <label htmlFor="pruneEMA" className="text-sm">Prune EMA weights</label>
          </div>
        </Accordion>
      </div>
    </div>
  );
};

export default CheckpointMergerTab;

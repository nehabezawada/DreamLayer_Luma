
import React, { useState } from 'react';
import Accordion from '@/components/Accordion';
import Slider from '@/components/Slider';
import SubTabNavigation from '@/components/SubTabNavigation';

const ExtrasTab = () => {
  const [activeSubTab, setActiveSubTab] = useState("upscale");
  
  const subtabs = [
    { id: "upscale", label: "Upscale", active: activeSubTab === "upscale" },
    { id: "process", label: "Post-processing", active: activeSubTab === "process" },
    { id: "batch", label: "Batch Process", active: activeSubTab === "batch" },
  ];

  const handleSubTabChange = (tabId: string) => {
    setActiveSubTab(tabId);
  };
  
  const renderSubTabContent = () => {
    switch (activeSubTab) {
      case "process":
        return (
          <div className="space-y-4 mb-4">
            <div className="mb-4">
              <label htmlFor="processor" className="mb-1 block text-sm">Processing Type:</label>
              <select id="processor" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="fix_face">Fix Faces</option>
                <option value="colorize">Colorize</option>
                <option value="remove_bg">Remove Background</option>
                <option value="enhance">Image Enhancement</option>
              </select>
            </div>
            
            <Slider
              min={0}
              max={100}
              defaultValue={75}
              label="Effect Strength"
            />
            
            <div className="flex items-center mb-4">
              <input type="checkbox" id="preserveColors" className="mr-2"/>
              <label htmlFor="preserveColors" className="text-sm">Preserve Original Colors</label>
            </div>
          </div>
        );
      case "batch":
        return (
          <div className="space-y-4 mb-4">
            <div className="mb-4 p-4 border-2 border-dashed border-border rounded-md text-center">
              <p className="text-muted-foreground mb-2">Drop multiple images here or click to browse</p>
              <button className="rounded-md bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground">
                Browse Files
              </button>
            </div>
            
            <div className="mb-4">
              <label htmlFor="batchOperation" className="mb-1 block text-sm">Batch Operation:</label>
              <select id="batchOperation" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="resize">Resize All</option>
                <option value="upscale">Upscale All</option>
                <option value="convert">Convert Format</option>
                <option value="enhance">Enhance All</option>
              </select>
            </div>
            
            <div className="mb-4">
              <label htmlFor="outputDir" className="mb-1 block text-sm">Output Directory:</label>
              <input
                id="outputDir"
                type="text"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="/output/batch_processed/"
              />
            </div>
          </div>
        );
      default:
        // "upscale" tab (default)
        return (
          <div className="space-y-4 mb-4">
            <div className="mb-4">
              <label htmlFor="upscaler" className="mb-1 block text-sm">Upscaler Model:</label>
              <select id="upscaler" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="esrgan">ESRGAN 4x</option>
                <option value="swinir">SwinIR 4x</option>
                <option value="real-esrgan">Real-ESRGAN 4x+</option>
                <option value="sd-upscaler">SD Upscaler</option>
              </select>
            </div>
            
            <Slider
              min={1}
              max={8}
              defaultValue={2}
              label="Upscale Factor"
            />
            
            <Slider
              min={0}
              max={100}
              defaultValue={50}
              label="Denoising Strength"
              sublabel="| Lower values preserve more details"
            />
            
            <div className="flex items-center mb-4">
              <input type="checkbox" id="faceRestoration" className="mr-2"/>
              <label htmlFor="faceRestoration" className="text-sm">Apply Face Restoration</label>
            </div>
            
            <div className="flex items-center">
              <input type="checkbox" id="colorCorrection" className="mr-2"/>
              <label htmlFor="colorCorrection" className="text-sm">Apply Color Correction</label>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="mb-4 grid gap-6 md:grid-cols-[2fr_1fr]">
      {/* Left Column - Controls */}
      <div className="space-y-4">
        <div className="flex flex-col">
          <div className="mb-[18px] flex flex-col space-y-2 sm:flex-row sm:items-center sm:justify-between sm:space-y-0">
            <h3 className="text-base font-medium">Image Post-Processing</h3>
            <div className="flex space-x-2">
              <button className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90">
                Process Image
              </button>
              <button className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground">
                Save Settings
              </button>
            </div>
          </div>
          
          <SubTabNavigation 
            className="mb-[18px]"
            tabs={subtabs} 
            onTabChange={handleSubTabChange}
          />
          
          <Accordion title="Input Image" number="1" defaultOpen={true}>
            <div className="mb-4 p-4 border-2 border-dashed border-border rounded-md text-center">
              <p className="text-muted-foreground mb-2">Drag & drop an image here or click to browse</p>
              <button className="rounded-md bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground">
                Browse Files
              </button>
            </div>
          </Accordion>
          
          <Accordion title={activeSubTab === "upscale" ? "Upscaling Options" : 
                      activeSubTab === "process" ? "Processing Options" : 
                      "Batch Options"} 
                     number="2" 
                     defaultOpen={true}>
            {renderSubTabContent()}
          </Accordion>
          
          <Accordion title="Output Settings" number="3">
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label htmlFor="outputWidth" className="mb-1 block text-sm">Width (px)</label>
                <input
                  id="outputWidth"
                  type="number"
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  placeholder="1024"
                />
              </div>
              <div>
                <label htmlFor="outputHeight" className="mb-1 block text-sm">Height (px)</label>
                <input
                  id="outputHeight"
                  type="number"
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  placeholder="1024"
                />
              </div>
            </div>
            
            <div className="mb-4">
              <label htmlFor="outputFormat" className="mb-1 block text-sm">Output Format:</label>
              <select id="outputFormat" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="png">PNG</option>
                <option value="jpg">JPG</option>
                <option value="webp">WebP</option>
              </select>
            </div>
          </Accordion>
        </div>
      </div>
      
      {/* Right Column - Preview */}
      <div>
        <div className="rounded-md border border-border p-4 h-[500px] flex items-center justify-center">
          <div className="text-center">
            <p className="text-muted-foreground">Processed image will display here</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExtrasTab;

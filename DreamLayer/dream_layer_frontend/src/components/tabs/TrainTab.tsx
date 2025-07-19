
import React, { useState } from 'react';
import Accordion from '@/components/Accordion';
import Slider from '@/components/Slider';
import SubTabNavigation from '@/components/SubTabNavigation';

const TrainTab = () => {
  const [activeSubTab, setActiveSubTab] = useState("dreambooth");
  
  const subtabs = [
    { id: "dreambooth", label: "DreamBooth", active: activeSubTab === "dreambooth" },
    { id: "lora", label: "LoRA", active: activeSubTab === "lora" },
    { id: "textual", label: "Textual Inversion", active: activeSubTab === "textual" },
    { id: "hypernetwork", label: "Hypernetwork", active: activeSubTab === "hypernetwork" }
  ];

  const handleSubTabChange = (tabId: string) => {
    setActiveSubTab(tabId);
  };
  
  const renderSubTabContent = () => {
    switch (activeSubTab) {
      case "lora":
        return (
          <>
            <div className="mb-4">
              <label htmlFor="loraName" className="mb-1 block text-sm font-medium">LoRA Name:</label>
              <input
                id="loraName"
                type="text"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="e.g., my_character_lora"
              />
            </div>
            
            <div className="mb-4 p-4 border-2 border-dashed border-border rounded-md text-center">
              <p className="text-muted-foreground mb-2">Upload training images (15-50 recommended)</p>
              <button className="rounded-md bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground">
                Browse Files
              </button>
            </div>
            
            <Slider
              min={1}
              max={32}
              defaultValue={8}
              label="Network Rank"
            />
            
            <Slider
              min={1}
              max={100}
              defaultValue={10}
              label="Network Alpha"
            />
          </>
        );
      case "textual":
        return (
          <>
            <div className="mb-4">
              <label htmlFor="embeddingName" className="mb-1 block text-sm font-medium">Embedding Name:</label>
              <input
                id="embeddingName"
                type="text"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="e.g., my_style_embedding"
              />
            </div>
            
            <div className="mb-4">
              <label htmlFor="initializationType" className="mb-1 block text-sm font-medium">Initialization:</label>
              <select id="initializationType" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="zero">Zero</option>
                <option value="text">From Text</option>
                <option value="image">From Image</option>
              </select>
            </div>
            
            <div className="mb-4 p-4 border-2 border-dashed border-border rounded-md text-center">
              <p className="text-muted-foreground mb-2">Upload training images (5-10 recommended)</p>
              <button className="rounded-md bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground">
                Browse Files
              </button>
            </div>
          </>
        );
      case "hypernetwork":
        return (
          <>
            <div className="mb-4">
              <label htmlFor="hypernetName" className="mb-1 block text-sm font-medium">Hypernetwork Name:</label>
              <input
                id="hypernetName"
                type="text"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="e.g., my_hypernetwork"
              />
            </div>
            
            <div className="mb-4">
              <label htmlFor="layerStructure" className="mb-1 block text-sm font-medium">Layer Structure:</label>
              <select id="layerStructure" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="1024, 1024, 1024">1024, 1024, 1024</option>
                <option value="1024, 512, 512">1024, 512, 512</option>
                <option value="512, 512, 512">512, 512, 512</option>
              </select>
            </div>
            
            <div className="mb-4 p-4 border-2 border-dashed border-border rounded-md text-center">
              <p className="text-muted-foreground mb-2">Upload training images (20-50 recommended)</p>
              <button className="rounded-md bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground">
                Browse Files
              </button>
            </div>
          </>
        );
      default:
        // "dreambooth" tab (default)
        return (
          <>
            <div className="mb-4">
              <label htmlFor="conceptName" className="mb-1 block text-sm font-medium">Concept Name:</label>
              <input
                id="conceptName"
                type="text"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="e.g., my_character_style"
              />
            </div>
            
            <div className="mb-4 p-4 border-2 border-dashed border-border rounded-md text-center">
              <p className="text-muted-foreground mb-2">Upload training images (10-20 recommended)</p>
              <button className="rounded-md bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground">
                Browse Files
              </button>
            </div>
            
            <div className="mb-4">
              <label htmlFor="classImages" className="mb-1 block text-sm font-medium">Class Images Dataset:</label>
              <select id="classImages" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="person">Person (FFHQ)</option>
                <option value="cat">Cat</option>
                <option value="dog">Dog</option>
                <option value="landscape">Landscape</option>
                <option value="custom">Custom...</option>
              </select>
            </div>
          </>
        );
    }
  };

  return (
    <div className="mb-4">
      <div className="flex flex-col">
        <div className="mb-[18px] flex flex-col space-y-2 sm:flex-row sm:items-center sm:justify-between sm:space-y-0">
          <h3 className="text-base font-medium">Model Training</h3>
          <div className="flex space-x-2">
            <button className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90">
              Start Training
            </button>
            <button className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground">
              Save Config
            </button>
          </div>
        </div>
        
        <SubTabNavigation 
          className="mb-[18px]"
          tabs={subtabs}
          onTabChange={handleSubTabChange}
        />
        
        <Accordion title="Training Data" number="1" defaultOpen={true}>
          {renderSubTabContent()}
        </Accordion>
        
        <Accordion title="Training Parameters" number="2" defaultOpen={true}>
          <div className="mb-4">
            <label htmlFor="baseModel" className="mb-1 block text-sm font-medium">Base Model:</label>
            <select id="baseModel" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
              <option value="sd15">StableDiffusion 1.5</option>
              <option value="sd21">StableDiffusion 2.1</option>
              <option value="sdxl">StableDiffusion XL</option>
            </select>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label htmlFor="instance_prompt" className="mb-1 block text-sm font-medium">Instance Prompt:</label>
              <input
                id="instance_prompt"
                type="text"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="a photo of sks dog"
              />
            </div>
            
            <div>
              <label htmlFor="class_prompt" className="mb-1 block text-sm font-medium">Class Prompt:</label>
              <input
                id="class_prompt"
                type="text"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="a photo of dog"
              />
            </div>
          </div>
          
          <Slider
            min={500}
            max={5000}
            defaultValue={1500}
            label="Training Steps"
          />
          
          <Slider
            min={1}
            max={100}
            defaultValue={8}
            label="Batch Size"
          />
          
          <Slider
            min={0.0001}
            max={0.01}
            defaultValue={0.0005}
            label="Learning Rate"
            inputWidth="w-24"
          />
        </Accordion>
        
        <Accordion title="Advanced Settings" number="3">
          <div className="mb-4">
            <label htmlFor="scheduler" className="mb-1 block text-sm font-medium">Scheduler:</label>
            <select id="scheduler" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
              <option value="constant">Constant</option>
              <option value="cosine">Cosine with Restarts</option>
              <option value="linear">Linear</option>
            </select>
          </div>
          
          <div className="mb-4 grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="resolution" className="mb-1 block text-sm font-medium">Resolution:</label>
              <input
                id="resolution"
                type="number"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="512"
                step="64"
                min="256"
              />
            </div>
            
            <div>
              <label htmlFor="prior_loss_weight" className="mb-1 block text-sm font-medium">Prior Loss Weight:</label>
              <input
                id="prior_loss_weight"
                type="number"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="1.0"
                step="0.1"
              />
            </div>
          </div>
          
          <div className="flex items-center mb-4">
            <input type="checkbox" id="with_prior_preservation" className="mr-2" defaultChecked />
            <label htmlFor="with_prior_preservation" className="text-sm">Use prior preservation</label>
          </div>
          
          <div className="flex items-center mb-4">
            <input type="checkbox" id="use_8bit_adam" className="mr-2" />
            <label htmlFor="use_8bit_adam" className="text-sm">Use 8-bit Adam optimizer</label>
          </div>
        </Accordion>
      </div>
    </div>
  );
};

export default TrainTab;


import React from 'react';
import Accordion from '@/components/Accordion';
import Slider from '@/components/Slider';

const ConfigurationsTab = () => {
  return (
    <div className="mb-4">
      <div className="flex flex-col">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-base font-medium">System Configurations</h3>
          <button className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90">
            Save Settings
          </button>
        </div>
        
        <Accordion title="UI Settings" number="1" defaultOpen={true}>
          <div className="mb-4">
            <label htmlFor="uiTheme" className="mb-1 block text-sm font-medium">UI Theme:</label>
            <select id="uiTheme" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
              <option value="system">System Default</option>
              <option value="light">Light Mode</option>
              <option value="dark">Dark Mode</option>
            </select>
          </div>
          
          <div className="mb-4">
            <label htmlFor="language" className="mb-1 block text-sm font-medium">Language:</label>
            <select id="language" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
              <option value="en">English</option>
              <option value="jp">日本語</option>
              <option value="de">Deutsch</option>
              <option value="fr">Français</option>
              <option value="es">Español</option>
            </select>
          </div>
          
          <div className="flex items-center mb-4">
            <input type="checkbox" id="quicksettings" className="mr-2" checked />
            <label htmlFor="quicksettings" className="text-sm">Show quick settings</label>
          </div>
          
          <div className="flex items-center mb-4">
            <input type="checkbox" id="progressInTitle" className="mr-2" checked />
            <label htmlFor="progressInTitle" className="text-sm">Show progress in title</label>
          </div>
        </Accordion>
        
        <Accordion title="Performance & Resources" number="2" defaultOpen={true}>
          <div className="mb-4">
            <label htmlFor="computeDevice" className="mb-1 block text-sm font-medium">Primary Compute Device:</label>
            <select id="computeDevice" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
              <option value="cuda">NVIDIA GPU (CUDA)</option>
              <option value="rocm">AMD GPU (ROCm)</option>
              <option value="mps">Apple Silicon (MPS)</option>
              <option value="cpu">CPU</option>
            </select>
          </div>
          
          <Slider
            min={0}
            max={100}
            defaultValue={85}
            label="VRAM Usage Target (%)"
            sublabel="| Lower to reduce memory usage"
          />
          
          <Slider
            min={1}
            max={16}
            defaultValue={2}
            label="Parallel Processing"
            sublabel="| Higher values use more GPU memory"
          />
          
          <div className="flex items-center mb-4">
            <input type="checkbox" id="xformers" className="mr-2" checked />
            <label htmlFor="xformers" className="text-sm">Use xFormers memory efficient attention</label>
          </div>
          
          <div className="flex items-center mb-4">
            <input type="checkbox" id="medvram" className="mr-2" />
            <label htmlFor="medvram" className="text-sm">Optimize for medium-low VRAM usage</label>
          </div>
        </Accordion>
        
        <Accordion title="Paths & Saving" number="3">
          <div className="mb-4">
            <label htmlFor="outputDir" className="mb-1 block text-sm font-medium">Output Directory:</label>
            <div className="flex">
              <input
                id="outputDir"
                type="text"
                className="flex-1 rounded-l-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="/path/to/outputs"
              />
              <button className="rounded-r-md bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground">
                Browse
              </button>
            </div>
          </div>
          
          <div className="mb-4">
            <label htmlFor="modelsDir" className="mb-1 block text-sm font-medium">Models Directory:</label>
            <div className="flex">
              <input
                id="modelsDir"
                type="text"
                className="flex-1 rounded-l-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="/path/to/models"
              />
              <button className="rounded-r-md bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground">
                Browse
              </button>
            </div>
          </div>
          
          <div className="mb-4">
            <label htmlFor="filenameFormat" className="mb-1 block text-sm font-medium">Filename Format:</label>
            <input
              id="filenameFormat"
              type="text"
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              placeholder="[model_name]_[seed]_[prompt_words]"
            />
            <p className="text-xs text-muted-foreground mt-1">
              Variables: [seed], [model_name], [prompt_words], [width], [height], [steps], [date], [time]
            </p>
          </div>
          
          <div className="flex items-center mb-4">
            <input type="checkbox" id="saveMetadata" className="mr-2" checked />
            <label htmlFor="saveMetadata" className="text-sm">Save generation parameters as metadata</label>
          </div>
        </Accordion>
        
        <Accordion title="Updates & Installation" number="4">
          <div className="flex items-center justify-between mb-4 p-4 bg-muted rounded-md">
            <div>
              <p className="font-medium">Current Version: v1.2.3</p>
              <p className="text-xs text-muted-foreground">Last checked: 4 hours ago</p>
            </div>
            <button className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground">
              Check for Updates
            </button>
          </div>
          
          <div className="mb-4">
            <label htmlFor="updateChannel" className="mb-1 block text-sm font-medium">Update Channel:</label>
            <select id="updateChannel" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
              <option value="stable">Stable</option>
              <option value="beta">Beta</option>
              <option value="nightly">Nightly (Experimental)</option>
            </select>
          </div>
          
          <div className="flex items-center mb-4">
            <input type="checkbox" id="autoUpdate" className="mr-2" />
            <label htmlFor="autoUpdate" className="text-sm">Enable automatic updates</label>
          </div>
        </Accordion>
      </div>
    </div>
  );
};

export default ConfigurationsTab;

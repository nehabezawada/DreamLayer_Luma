
import React from 'react';
import Accordion from '@/components/Accordion';

const installedExtensions = [
  { 
    id: "controlnet",
    name: "ControlNet v1.1.411", 
    author: "Mikubill",
    description: "Adds support for ControlNet conditioning to generate images guided by depth, canny edge, segmentation, etc."
  },
  { 
    id: "adetailer", 
    name: "ADetailer",
    author: "Bing-su",
    description: "Face and body detailing/restoration tool"
  },
  { 
    id: "regional-prompter", 
    name: "Regional Prompter",
    author: "hako-mikan",
    description: "Enables different prompts for different regions of the image"
  }
];

const availableExtensions = [
  { 
    id: "ultimate-upscale", 
    name: "Ultimate SD Upscale",
    author: "Coyote-A",
    description: "Ultimate upscaler for SD"
  },
  { 
    id: "image-browser", 
    name: "Image Browser",
    author: "yfszzx",
    description: "Browse and manage your generated images"
  },
  { 
    id: "wildcards", 
    name: "Wildcards",
    author: "AUTOMATIC1111",
    description: "Use wildcards in your prompts"
  }
];

const ExtensionsTab = () => {
  return (
    <div className="mb-4">
      <div className="flex flex-col">
        <div className="mb-3 flex flex-col space-y-2 sm:flex-row sm:items-center sm:justify-between sm:space-y-0">
          <h3 className="text-base font-medium">Extensions Manager</h3>
          <div className="flex space-x-2">
            <button className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90">
              Check for Updates
            </button>
            <button className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground">
              Reload UI
            </button>
          </div>
        </div>
        
        <div className="mb-4">
          <div className="relative">
            <input
              type="text"
              className="w-full rounded-md border border-input bg-background px-3 py-2 pl-10 text-sm"
              placeholder="Search extensions..."
            />
            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
              <svg className="w-4 h-4 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
          </div>
        </div>
        
        <Accordion title="Installed Extensions" number="1" defaultOpen={true}>
          <div className="space-y-4">
            {installedExtensions.map((ext) => (
              <div key={ext.id} className="rounded-md border border-border p-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                  <div>
                    <h4 className="font-medium">{ext.name}</h4>
                    <p className="text-xs text-muted-foreground">Author: {ext.author}</p>
                  </div>
                  <div className="flex space-x-2">
                    <button className="rounded-md bg-secondary px-3 py-1 text-xs font-medium text-secondary-foreground">
                      Settings
                    </button>
                    <button className="rounded-md border border-destructive bg-transparent px-3 py-1 text-xs font-medium text-destructive hover:bg-destructive/10">
                      Uninstall
                    </button>
                  </div>
                </div>
                <p className="text-sm">{ext.description}</p>
              </div>
            ))}
          </div>
        </Accordion>
        
        <Accordion title="Available Extensions" number="2" defaultOpen={true}>
          <div className="space-y-4">
            {availableExtensions.map((ext) => (
              <div key={ext.id} className="rounded-md border border-border p-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                  <div>
                    <h4 className="font-medium">{ext.name}</h4>
                    <p className="text-xs text-muted-foreground">Author: {ext.author}</p>
                  </div>
                  <button className="rounded-md bg-primary px-3 py-1 text-xs font-medium text-primary-foreground">
                    Install
                  </button>
                </div>
                <p className="text-sm">{ext.description}</p>
              </div>
            ))}
          </div>
        </Accordion>
        
        <Accordion title="Extension Settings" number="3">
          <div className="mb-4">
            <label htmlFor="extensionSource" className="mb-1 block text-sm font-medium">Extension Source:</label>
            <select id="extensionSource" className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
              <option value="index">Main Extension Index</option>
              <option value="custom">Custom Extension Index</option>
            </select>
          </div>
          
          <div className="mb-4">
            <label className="mb-1 block text-sm font-medium">Install From URL:</label>
            <div className="flex">
              <input
                type="text"
                className="flex-1 rounded-l-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="https://github.com/username/extension"
              />
              <button className="rounded-r-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground">
                Install
              </button>
            </div>
          </div>
          
          <div className="flex items-center mb-4">
            <input type="checkbox" id="extensionUpdates" className="mr-2" checked />
            <label htmlFor="extensionUpdates" className="text-sm">Check for extension updates on startup</label>
          </div>
          
          <div className="flex items-center mb-4">
            <input type="checkbox" id="devMode" className="mr-2" />
            <label htmlFor="devMode" className="text-sm">Developer mode</label>
          </div>
        </Accordion>
      </div>
    </div>
  );
};

export default ExtensionsTab;

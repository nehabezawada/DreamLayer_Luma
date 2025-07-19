
import React from 'react';
import Accordion from '@/components/Accordion';

const PNGInfoPage = () => {
  return (
    <div className="flex justify-center">
      <div className="w-full max-w-4xl">
        {/* Centered Content */}
        <div className="space-y-4">
          <div className="flex flex-col">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-base font-medium">PNG Info Extraction</h3>
              <button className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90">
                Extract Info
              </button>
            </div>
            
            <Accordion title="Image Upload" number="1" defaultOpen={true}>
              <div className="mb-4 p-4 border-2 border-dashed border-border rounded-md text-center">
                <p className="text-muted-foreground mb-2">Drag & drop a PNG image here or click to browse</p>
                <p className="text-xs text-muted-foreground mb-4">Only PNG images with embedded generation data will provide full information</p>
                <button className="rounded-md bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground">
                  Browse Files
                </button>
              </div>
            </Accordion>
            
            <Accordion title="Extracted Information" number="2" defaultOpen={true}>
              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-medium mb-1">Prompt:</h4>
                  <div className="rounded-md bg-muted p-3 text-sm min-h-[100px] whitespace-pre-wrap">
                    <p className="text-muted-foreground italic">Prompt information will appear here...</p>
                  </div>
                </div>
                
                <div>
                  <h4 className="text-sm font-medium mb-1">Negative Prompt:</h4>
                  <div className="rounded-md bg-muted p-3 text-sm min-h-[80px] whitespace-pre-wrap">
                    <p className="text-muted-foreground italic">Negative prompt information will appear here...</p>
                  </div>
                </div>
                
                <div>
                  <h4 className="text-sm font-medium mb-1">Generation Settings:</h4>
                  <div className="rounded-md bg-muted p-3 text-sm">
                    <p className="text-muted-foreground italic">Settings information will appear here...</p>
                  </div>
                </div>
              </div>
              
              <div className="mt-4 flex gap-2">
                <button className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground">
                  Copy to Clipboard
                </button>
                <button className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground">
                  Send to Txt2Img
                </button>
                <button className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground">
                  Send to Img2Img
                </button>
              </div>
            </Accordion>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PNGInfoPage;

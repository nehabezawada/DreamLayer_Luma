
import React from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";

const XYPlotSettings: React.FC = () => {
  return (
    <>
      <div className="my-5 space-y-4">
        <div className="mb-2 text-sm font-medium">b) X/Y/Z Plot</div>
        
        {/* X Type and Values Row */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium text-[#64748B] mb-2 block">X Type:</label>
            <Select defaultValue="steps">
              <SelectTrigger className="w-full mt-1">
                <SelectValue placeholder="Select X axis type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="prompt">Prompt S/R</SelectItem>
                <SelectItem value="steps">Steps</SelectItem>
                <SelectItem value="cfg">CFG Scale</SelectItem>
                <SelectItem value="sampler">Sampler</SelectItem>
                <SelectItem value="seed">Seed</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div>
            <label htmlFor="x-values" className="text-sm font-medium text-[#64748B] mb-2 block">X Values:</label>
            <Textarea 
              id="x-values" 
              className="mt-1" 
              placeholder="Enter comma-separated values for X axis"
            />
            <div className="text-xs text-muted-foreground mt-1">Separate multiple values with commas</div>
          </div>
        </div>
        
        {/* Y Type and Values Row */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium text-[#64748B] mb-2 block">Y Type:</label>
            <Select defaultValue="seed">
              <SelectTrigger className="w-full mt-1">
                <SelectValue placeholder="Select Y axis type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="prompt">Prompt S/R</SelectItem>
                <SelectItem value="steps">Steps</SelectItem>
                <SelectItem value="cfg">CFG Scale</SelectItem>
                <SelectItem value="sampler">Sampler</SelectItem>
                <SelectItem value="seed">Seed</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div>
            <label htmlFor="y-values" className="text-sm font-medium text-[#64748B] mb-2 block">Y Values:</label>
            <Textarea 
              id="y-values" 
              className="mt-1" 
              placeholder="Enter comma-separated values for Y axis"
            />
            <div className="text-xs text-muted-foreground mt-1">Separate multiple values with commas</div>
          </div>
        </div>
        
        {/* Z Type and Values Row */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium text-[#64748B] mb-2 block">Z Type:</label>
            <Select defaultValue="cfg">
              <SelectTrigger className="w-full mt-1">
                <SelectValue placeholder="Select Z axis type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="prompt">Prompt S/R</SelectItem>
                <SelectItem value="steps">Steps</SelectItem>
                <SelectItem value="cfg">CFG Scale</SelectItem>
                <SelectItem value="sampler">Sampler</SelectItem>
                <SelectItem value="seed">Seed</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div>
            <label htmlFor="z-values" className="text-sm font-medium text-[#64748B] mb-2 block">Z Values:</label>
            <Textarea 
              id="z-values" 
              className="mt-1" 
              placeholder="Enter comma-separated values for Z axis"
            />
            <div className="text-xs text-muted-foreground mt-1">Separate multiple values with commas</div>
          </div>
        </div>
      </div>
      
      {/* Swap buttons - With new title */}
      <div className="my-4">
        <div className="mb-2 text-sm font-medium">c) Swap Axes Options</div>
        <div className="flex flex-row space-x-2 mb-5">
          <Button variant="outline" size="sm">
            Swap X/Y axes
          </Button>
          <Button variant="outline" size="sm">
            Swap Y/Z axes
          </Button>
          <Button variant="outline" size="sm">
            Swap X/Z axes
          </Button>
        </div>
      </div>
      
      {/* Grid options - With updated title */}
      <div className="my-4">
        <div className="mb-2 text-sm font-medium">d) Grid Display Settings</div>
        <div className="flex items-center gap-2 mb-3">
          <Checkbox id="draw-legend" />
          <label htmlFor="draw-legend" className="text-sm">Draw legend</label>
        </div>
        
        <div className="flex items-center gap-2 mb-3">
          <Checkbox id="include-sub-images" defaultChecked />
          <label htmlFor="include-sub-images" className="text-sm">Include sub-images</label>
        </div>
        
        <div className="flex items-center gap-2 mb-3">
          <Checkbox id="include-sub-grids" />
          <label htmlFor="include-sub-grids" className="text-sm">Include sub-grids</label>
        </div>
        
        <div className="flex items-center gap-2 mb-3">
          <Checkbox id="keep-1-for-seeds" />
          <label htmlFor="keep-1-for-seeds" className="text-sm">Keep-1 for seeds</label>
        </div>
      </div>
    </>
  );
};

export default XYPlotSettings;

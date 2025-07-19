import React from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import ControlNetImageUpload from "./ControlNetImageUpload";
import ControlNetPreview from "./ControlNetPreview";

interface ControlNetInputTabsProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  allowPreview: boolean;
  uploadIndependentControl: boolean;
  onImageChange?: (image: { url: string, file: File, filename?: string } | null) => void;
  unitIndex?: number; // Add unit index
}

const ControlNetInputTabs: React.FC<ControlNetInputTabsProps> = ({ 
  activeTab, 
  setActiveTab, 
  allowPreview,
  uploadIndependentControl,
  onImageChange,
  unitIndex = 0
}) => {

  return (
    <div className="p-2">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="single">Single Image</TabsTrigger>
          <TabsTrigger value="batch">Batch</TabsTrigger>
        </TabsList>
        
        <TabsContent value="single" className="pt-3">
          {!uploadIndependentControl && (
            <div className={allowPreview ? "grid grid-cols-2 gap-4" : "grid grid-cols-1"}>
              <div className="space-y-2">
                <ControlNetImageUpload onImageChange={onImageChange} unitIndex={unitIndex} />
              </div>
              
              {allowPreview && (
                <div className="space-y-2">
                  <ControlNetPreview />
                </div>
              )}
            </div>
          )}
          
          {uploadIndependentControl && (
            <div className="mt-3 space-y-2">
              <div className="text-xs text-muted-foreground">
                Set the preprocessor to [invert] if your image has white background and black lines.
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox id="loopback" />
                <Label htmlFor="loopback" className="text-sm text-muted-foreground font-normal">
                  [Loopback] Automatically send generated images to this ControlNet unit
                </Label>
              </div>
            </div>
          )}
          
          {!uploadIndependentControl && (
            <div className="mt-3 space-y-2">
              <div className="text-xs text-muted-foreground">
                Set the preprocessor to [invert] if your image has white background and black lines.
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox id="loopback" />
                <Label htmlFor="loopback" className="text-sm text-muted-foreground font-normal">
                  [Loopback] Automatically send generated images to this ControlNet unit
                </Label>
              </div>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="batch" className="pt-3">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="font-medium text-sm">Input Directory - Batch image processing</div>
              <Button 
                size="sm" 
                className="text-xs"
                style={{ height: '34px' }}
              >
                Browse Batch Files
              </Button>
            </div>
            <Textarea 
              placeholder="Specify a folder containing reference images"
              className="min-h-[100px]"
            />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ControlNetInputTabs;

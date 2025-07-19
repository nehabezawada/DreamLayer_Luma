
import React from "react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface ControlNetTogglesProps {
  unitIndex: number;
  enabled: boolean;
  setEnabled: (enabled: boolean) => void;
  lowVram: boolean;
  setLowVram: (lowVram: boolean) => void;
  pixelPerfect: boolean;
  setPixelPerfect: (pixelPerfect: boolean) => void;
  allowPreview: boolean;
  setAllowPreview: (allowPreview: boolean) => void;
  effectiveRegionMask: boolean;
  setEffectiveRegionMask: (effectiveRegionMask: boolean) => void;
  uploadIndependentControl: boolean;
  setUploadIndependentControl: (uploadIndependentControl: boolean) => void;
  selectedControlType: string;
}

const ControlNetToggles: React.FC<ControlNetTogglesProps> = ({
  unitIndex,
  enabled,
  setEnabled,
  lowVram,
  setLowVram,
  pixelPerfect,
  setPixelPerfect,
  allowPreview,
  setAllowPreview,
  effectiveRegionMask,
  setEffectiveRegionMask,
  uploadIndependentControl,
  setUploadIndependentControl,
  selectedControlType,
}) => {
  // Control types that should show the Effective Region Mask toggle
  const controlTypesWithRegionMask = ["ip-adapter", "t2i-adapter", "inpaint", "revision"];
  const shouldShowRegionMask = controlTypesWithRegionMask.includes(selectedControlType);

  return (
    <div className="p-4 border-t border-input">
      <Label className="text-sm mb-3 block">Processing Options</Label>
      <div className="flex flex-wrap gap-4">
        <div className="flex items-center space-x-2">
          <Switch 
            id={`unit-${unitIndex}-enabled`}
            checked={enabled}
            onCheckedChange={setEnabled}
          />
          <Label htmlFor={`unit-${unitIndex}-enabled`} className="text-sm">Enable</Label>
        </div>
        
        <div className="flex items-center space-x-2">
          <Switch 
            id={`unit-${unitIndex}-lowvram`}
            checked={lowVram}
            onCheckedChange={setLowVram}
          />
          <Label htmlFor={`unit-${unitIndex}-lowvram`} className="text-sm">Low VRAM</Label>
        </div>
        
        <div className="flex items-center space-x-2">
          <Switch 
            id={`unit-${unitIndex}-pixelperfect`}
            checked={pixelPerfect}
            onCheckedChange={setPixelPerfect}
          />
          <Label htmlFor={`unit-${unitIndex}-pixelperfect`} className="text-sm">Pixel Perfect</Label>
        </div>
        
        <div className="flex items-center space-x-2">
          <Switch 
            id={`unit-${unitIndex}-preview`}
            checked={allowPreview}
            onCheckedChange={setAllowPreview}
          />
          <Label htmlFor={`unit-${unitIndex}-preview`} className="text-sm">Allow Preview</Label>
        </div>
        
        <div className="flex items-center space-x-2">
          <Switch 
            id={`unit-${unitIndex}-independent-control`}
            checked={uploadIndependentControl}
            onCheckedChange={setUploadIndependentControl}
          />
          <Label htmlFor={`unit-${unitIndex}-independent-control`} className="text-sm">Upload Independent Control Image</Label>
        </div>
        
        {shouldShowRegionMask && (
          <div className="flex items-center space-x-2">
            <Switch 
              id={`unit-${unitIndex}-region-mask`}
              checked={effectiveRegionMask}
              onCheckedChange={setEffectiveRegionMask}
            />
            <Label htmlFor={`unit-${unitIndex}-region-mask`} className="text-sm">Effective Region Mask</Label>
          </div>
        )}
      </div>
    </div>
  );
};

export default ControlNetToggles;

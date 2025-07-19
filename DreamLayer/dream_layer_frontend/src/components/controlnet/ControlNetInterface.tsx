import React, { useCallback } from "react";
import ControlNetUnit from "./ControlNetUnit";
import { Button } from "@/components/ui/button";
import { X, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { ControlNetRequest, ControlNetUnit as ControlNetUnitType } from "@/types/controlnet";
import useControlNetStore from "@/stores/useControlNetStore";

interface ControlNetInterfaceProps {
  isImg2ImgTab?: boolean;
  onControlNetChange?: (controlNet: ControlNetRequest) => void;
}

const ControlNetInterface = ({ 
  isImg2ImgTab = false,
  onControlNetChange 
}: ControlNetInterfaceProps) => {
  const {
    activeUnits,
    selectedUnitIndex,
    controlNetUnits,
    setSelectedUnitIndex,
    updateControlNetUnit,
    addUnit,
    removeUnit
  } = useControlNetStore();

  // Update parent component when controlnet configuration changes
  const updateControlNetRequest = useCallback(() => {
    if (!onControlNetChange) return;
      // Always build config from the latest store state
    const state = useControlNetStore.getState();
    const controlNetUnits = state.controlNetUnits;

    console.log('[ControlNetInterface] updateControlNetRequest called');
    console.log('[ControlNetInterface] Current controlNetUnits:', controlNetUnits);
    
    const enabledUnits = Object.values(controlNetUnits)
      .filter(unit => unit.enabled)
      .map(unit => {
        console.log(`[ControlNetInterface] Processing unit ${unit.unit_index}:`, {
          enabled: unit.enabled,
          hasInputImage: unit.input_image !== null && unit.input_image !== undefined,
          inputImageType: typeof unit.input_image,
          inputImageLength: unit.input_image ? unit.input_image.length : 'N/A'
        });
        return { ...unit, input_image: unit.input_image || null };
      });
    
    console.log('[ControlNetInterface] Final enabled units:', enabledUnits);
    console.log('[ControlNetInterface] Calling onControlNetChange with:', {
      enabled: enabledUnits.length > 0,
      units: enabledUnits
    });
    
    onControlNetChange({
      enabled: enabledUnits.length > 0,
      units: enabledUnits
    });
  }, [controlNetUnits, onControlNetChange]);

  const handleTabClick = (unitIndex: number) => {
    console.log('[ControlNetInterface] Tab clicked. Selected unit index:', unitIndex);
    setSelectedUnitIndex(unitIndex);
  };

  const handleUnitChange = (unitIndex: number, updatedUnit: Partial<ControlNetUnitType>) => {
    console.log('[ControlNetInterface] Unit changed:', { 
      unitIndex, 
      updatedUnit
    });
    
    // Check if this is an input_image update
    if ('input_image' in updatedUnit) {
      console.log('[ControlNetInterface] Input image update detected:');
      console.log('[ControlNetInterface] - Old value:', controlNetUnits[unitIndex]?.input_image);
      console.log('[ControlNetInterface] - New value:', updatedUnit.input_image);
      console.log('[ControlNetInterface] - Type:', typeof updatedUnit.input_image);
    }
    
    updateControlNetUnit(unitIndex, updatedUnit);
    
    // // Add a small delay to ensure the store is updated before building the request
    setTimeout(() => {
      console.log('[ControlNetInterface] Delayed updateControlNetRequest call');
      console.log('[ControlNetInterface] Current unit state after update:', controlNetUnits[unitIndex]);
      updateControlNetRequest();
    }, 50);

  };

  // Count enabled units
  const enabledUnitsCount = Object.values(controlNetUnits).filter(unit => unit.enabled).length;

  console.log('[ControlNetInterface] Render: activeUnits=', activeUnits, 'selectedUnitIndex=', selectedUnitIndex, 'enabledUnitsCount=', enabledUnitsCount);

  return (
    <div className="space-y-4 p-3">
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center">
          <div className="font-medium mr-3" style={{ fontSize: '16px', marginRight: '12px' }}>ControlNet</div>
          {enabledUnitsCount > 0 && (
            <div className="bg-secondary text-secondary-foreground text-sm rounded-md" style={{
              paddingTop: '4px',
              paddingBottom: '4px',
              paddingLeft: '10px',
              paddingRight: '10px',
              fontSize: '14px',
              marginLeft: '0px'
            }}>
              {enabledUnitsCount} Unit{enabledUnitsCount !== 1 ? 's' : ''}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          {activeUnits.length < 10 && (
            <Button 
              onClick={addUnit}
              className="bg-primary text-primary-foreground"
              size="sm"
            >
              Add Unit
            </Button>
          )}
          <button 
            onClick={() => {
              activeUnits.forEach(unitIndex => {
                const unit = controlNetUnits[unitIndex];
                console.log(`Unit ${unitIndex}:`, {
                  enabled: unit.enabled,
                  input_image: unit.input_image,
                  input_image_type: typeof unit.input_image,
                  input_image_length: unit.input_image ? unit.input_image.length : 'N/A'
                });
              });
            }}
            className="rounded-md border border-input bg-background px-2 py-1 text-xs font-medium transition-colors hover:bg-accent hover:text-accent-foreground"
          >
            Debug
          </button>
          <button className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground">
            Save Settings
          </button>
        </div>
      </div>

      {/* Unit Tabs */}
      <div className="flex flex-wrap gap-2 mb-4">
        {activeUnits.map((unitIndex) => (
          <div key={`unit-${unitIndex}`} className="relative group">
            <button
              onClick={() => handleTabClick(unitIndex)}
              className={cn(
                "px-5 py-2 rounded-full text-sm font-medium transition-all flex items-center gap-1",
                selectedUnitIndex === unitIndex 
                  ? "bg-primary text-primary-foreground shadow-sm" 
                  : "bg-secondary text-secondary-foreground hover:bg-secondary/80 border border-border"
              )}
            >
              {controlNetUnits[unitIndex]?.enabled && (
                <Check className="h-3 w-3" />
              )}
              ControlNet Unit {unitIndex}
              {unitIndex !== 0 && (
                <X 
                  className="h-3 w-3 opacity-70 hover:opacity-100 ml-1" 
                  onClick={(e) => {
                    e.stopPropagation();
                    removeUnit(unitIndex);
                  }}
                />
              )}
            </button>
          </div>
        ))}
      </div>
      
      {/* Selected Unit Content */}
      <div>
        {activeUnits.map((unitIndex) => (
          <div 
            key={`content-${unitIndex}`} 
            className={unitIndex === selectedUnitIndex ? "block" : "hidden"}
          >
            <ControlNetUnit 
              unitIndex={unitIndex} 
              unit={controlNetUnits[unitIndex]}
              onUnitChange={(updatedUnit) => handleUnitChange(unitIndex, updatedUnit)}
              onRemove={() => activeUnits.length > 1 && removeUnit(unitIndex)}
              defaultUploadIndependentControl={isImg2ImgTab}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default ControlNetInterface;

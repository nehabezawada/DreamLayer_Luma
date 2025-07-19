import React from "react";
import ControlNetHeader from "./ControlNetHeader";
import ControlNetInputTabs from "./ControlNetInputTabs";
import ControlNetToggles from "./ControlNetToggles";
import ControlTypeSelector from "./ControlTypeSelector";
import PreprocessorModelSelector from "./PreprocessorModelSelector";
import ThresholdControls from "./ThresholdControls";
import ControlParameters from "./ControlParameters";
import ControlModeSelector from "./ControlModeSelector";
import { ControlNetUnit as ControlNetUnitType } from "@/types/controlnet";

interface ControlNetUnitProps {
  unitIndex: number;
  unit: ControlNetUnitType;
  onUnitChange: (updatedUnit: Partial<ControlNetUnitType>) => void;
  onRemove: () => void;
  defaultUploadIndependentControl?: boolean;
}

const ControlNetUnit: React.FC<ControlNetUnitProps> = ({ 
  unitIndex, 
  unit,
  onUnitChange,
  onRemove, 
  defaultUploadIndependentControl = false
}) => {
  // Handle state changes
  const handleChange = (changes: Partial<ControlNetUnitType>) => {
    onUnitChange(changes);
  };

  const handleImageChange = (image: { url: string, file: File, filename?: string } | null) => {
    console.log('ControlNetUnit: Image changed:', image ? 'Image received' : 'No image');
    console.log('ControlNetUnit: Image details:', image);
    
    if (image?.file) {
      if (image.filename) {
        console.log('ControlNetUnit: Using server filename:', image.filename);
        handleChange({ input_image: image.filename });
      } else {
        // Fallback to base64 conversion if upload failed
        const reader = new FileReader();
        reader.onload = () => {
          const base64Data = reader.result as string;
          handleChange({ input_image: base64Data });
        };
        reader.onerror = (error) => {
          console.error('ControlNetUnit: Error reading file:', error);
        };
        reader.readAsDataURL(image.file);
      }
    } else {
      handleChange({ input_image: null });
    }
  };

  return (
    <div className="border border-input rounded-md bg-card">
      {/* Unit header */}
      <ControlNetHeader 
        unitIndex={unitIndex} 
        selectedControlType={unit.control_type} 
      />
      
      {/* Input Tabs */}
      <ControlNetInputTabs 
        activeTab={unit.input_mode === "batch" ? "batch" : "single"}
        setActiveTab={(tab) => handleChange({ input_mode: tab as "single" | "batch" })}
        allowPreview={unit.allow_preview}
        uploadIndependentControl={unit.upload_independent_control}
        onImageChange={handleImageChange}
        unitIndex={unitIndex}
      />

      {/* Unit Toggles Row */}
      <ControlNetToggles 
        unitIndex={unitIndex}
        enabled={unit.enabled}
        setEnabled={(enabled) => handleChange({ enabled })}
        lowVram={unit.low_vram}
        setLowVram={(low_vram) => handleChange({ low_vram })}
        pixelPerfect={unit.pixel_perfect}
        setPixelPerfect={(pixel_perfect) => handleChange({ pixel_perfect })}
        allowPreview={unit.allow_preview}
        setAllowPreview={(allow_preview) => handleChange({ allow_preview })}
        effectiveRegionMask={unit.effective_region_mask}
        setEffectiveRegionMask={(effective_region_mask) => handleChange({ effective_region_mask })}
        uploadIndependentControl={unit.upload_independent_control}
        setUploadIndependentControl={(upload_independent_control) => handleChange({ upload_independent_control })}
        selectedControlType={unit.control_type}
      />
      
      {/* Control Type Selector */}
      <ControlTypeSelector 
        selectedControlType={unit.control_type}
        onControlTypeChange={(control_type) => handleChange({ control_type })}
      />
      
      {/* Preprocessor and Model Selectors */}
      <PreprocessorModelSelector 
        unitIndex={unitIndex}
        selectedPreprocessor={unit.preprocessor}
        setSelectedPreprocessor={(preprocessor) => handleChange({ preprocessor })}
        selectedModel={unit.model}
        setSelectedModel={(model) => handleChange({ model })}
      />
      
      {/* Threshold controls - only show for preprocessors that support them */}
      <ThresholdControls 
        unitIndex={unitIndex}
        selectedPreprocessor={unit.preprocessor}
        thresholdA={unit.threshold_a ?? 100}
        setThresholdA={(threshold_a) => handleChange({ threshold_a })}
        thresholdB={unit.threshold_b ?? 200}
        setThresholdB={(threshold_b) => handleChange({ threshold_b })}
      />
      
      {/* Control Parameters */}
      <ControlParameters 
        unitIndex={unitIndex}
        weight={unit.weight}
        setWeight={(weight) => handleChange({ weight })}
        guidanceStart={unit.guidance_start}
        setGuidanceStart={(guidance_start) => handleChange({ guidance_start })}
        guidanceEnd={unit.guidance_end}
        setGuidanceEnd={(guidance_end) => handleChange({ guidance_end })}
        resolution={unit.resolution}
        setResolution={(resolution) => handleChange({ resolution })}
      />
      
      {/* Control Mode and Resize Mode */}
      <ControlModeSelector 
        controlMode={unit.control_mode}
        setControlMode={(control_mode) => handleChange({ control_mode })}
        resizeMode={unit.resize_mode}
        setResizeMode={(resize_mode) => handleChange({ resize_mode })}
      />
    </div>
  );
};

export default ControlNetUnit;

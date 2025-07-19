
import React from "react";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Card } from "@/components/ui/card";
import { controlModes, resizeModes } from "./controlnet-data";

interface ControlModeSelectorProps {
  controlMode: string;
  setControlMode: (mode: string) => void;
  resizeMode: string;
  setResizeMode: (mode: string) => void;
}

const ControlModeSelector: React.FC<ControlModeSelectorProps> = ({
  controlMode,
  setControlMode,
  resizeMode,
  setResizeMode,
}) => {
  return (
    <div className="p-4 border-t border-input">
      <div className="mb-4">
        <Label className="text-sm mb-3 block">Control Mode</Label>
        <RadioGroup 
          value={controlMode} 
          onValueChange={(value) => value && setControlMode(value)} 
          className="grid grid-cols-3 gap-3"
        >
          {controlModes.map((mode) => (
            <div key={mode.id} className="relative">
              <Card 
                className={`flex items-center gap-3 p-3 cursor-pointer border ${
                  controlMode === mode.id 
                    ? "border-blue-500 bg-blue-50/50" 
                    : "border-gray-200"
                }`}
              >
                <RadioGroupItem id={`control-mode-${mode.id}`} value={mode.id} className="h-4 w-4" />
                <label 
                  htmlFor={`control-mode-${mode.id}`} 
                  className="text-sm font-medium cursor-pointer w-full"
                >
                  {mode.label}
                </label>
              </Card>
            </div>
          ))}
        </RadioGroup>
      </div>
      
      <div>
        <Label className="text-sm mb-3 block">Resize Mode</Label>
        <RadioGroup 
          value={resizeMode} 
          onValueChange={(value) => value && setResizeMode(value)} 
          className="grid grid-cols-3 gap-3"
        >
          {resizeModes.map((mode) => (
            <div key={mode.id} className="relative">
              <Card 
                className={`flex items-center gap-3 p-3 cursor-pointer border ${
                  resizeMode === mode.id 
                    ? "border-blue-500 bg-blue-50/50" 
                    : "border-gray-200"
                }`}
              >
                <RadioGroupItem id={`resize-mode-${mode.id}`} value={mode.id} className="h-4 w-4" />
                <label 
                  htmlFor={`resize-mode-${mode.id}`} 
                  className="text-sm font-medium cursor-pointer w-full"
                >
                  {mode.label}
                </label>
              </Card>
            </div>
          ))}
        </RadioGroup>
      </div>
    </div>
  );
};

export default ControlModeSelector;


import React from "react";
import { Label } from "@/components/ui/label";
import { controlTypes } from "./controlnet-data";

interface ControlTypeSelectorProps {
  selectedControlType: string;
  onControlTypeChange: (type: string) => void;
}

const ControlTypeSelector: React.FC<ControlTypeSelectorProps> = ({ 
  selectedControlType, 
  onControlTypeChange 
}) => {
  return (
    <div className="p-4 border-t border-input">
      <Label className="text-sm mb-3 block">Control Type</Label>
      <div className="flex flex-wrap gap-1 overflow-x-auto pb-2">
        {controlTypes.map((type) => (
          <button
            key={type.id}
            className={`flex items-center px-3 py-1 rounded-full text-xs ${
              selectedControlType === type.id
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-secondary-foreground hover:bg-accent"
            }`}
            onClick={() => onControlTypeChange(type.id)}
          >
            {type.label}
          </button>
        ))}
      </div>
    </div>
  );
};

export default ControlTypeSelector;

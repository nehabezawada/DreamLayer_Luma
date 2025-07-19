
import React, { useState } from "react";
import { Trash2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface ControlNetHeaderProps {
  unitIndex: number;
  selectedControlType: string;
}

const ControlNetHeader: React.FC<ControlNetHeaderProps> = ({ unitIndex, selectedControlType }) => {
  const [selectedPreset, setSelectedPreset] = useState<string>("");

  const handlePresetChange = (value: string) => {
    setSelectedPreset(value);
  };

  const handleDeletePreset = () => {
    setSelectedPreset("");
  };

  const handleRefreshPresets = () => {
    console.log("Refreshing presets...");
  };

  const showDeleteButton = selectedPreset && selectedPreset !== "none" && selectedPreset !== "";

  return (
    <div className="p-2 rounded-t-md border-b border-input bg-card">
      <div className="flex items-center justify-between">
        <h4 className="font-medium text-foreground" style={{ fontSize: '14px' }}>
          ControlNet Unit {unitIndex} {selectedControlType && `[${selectedControlType.charAt(0).toUpperCase() + selectedControlType.slice(1)}]`}
        </h4>
        <div className="flex items-center gap-2">
          <Select value={selectedPreset} onValueChange={handlePresetChange}>
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Select Saved Preset" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">None</SelectItem>
              <SelectItem value="preset1">Preset 1</SelectItem>
              <SelectItem value="preset2">Preset 2</SelectItem>
              <SelectItem value="preset3">Preset 3</SelectItem>
            </SelectContent>
          </Select>
          <Button 
            size="sm" 
            variant="outline" 
            className="w-[30px] h-[30px] p-0" 
            onClick={handleRefreshPresets}
          >
            <RefreshCw className="w-4 h-4" />
          </Button>
          {showDeleteButton && (
            <Button 
              size="sm" 
              variant="outline" 
              className="w-[30px] h-[30px] p-0" 
              onClick={handleDeletePreset}
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ControlNetHeader;


import React from "react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import Slider from "@/components/Slider";

interface PromptMatrixSettingsProps {
  promptTarget: string;
  setPromptTarget: (value: string) => void;
  separator: string;
  setSeparator: (value: string) => void;
  imageSpacing: number;
  setImageSpacing: (value: number) => void;
  variablePartsAtStart: boolean;
  setVariablePartsAtStart: (value: boolean) => void;
  differentSeedForEach: boolean;
  setDifferentSeedForEach: (value: boolean) => void;
}

const PromptMatrixSettings: React.FC<PromptMatrixSettingsProps> = ({
  promptTarget,
  setPromptTarget,
  separator,
  setSeparator,
  imageSpacing,
  setImageSpacing,
  variablePartsAtStart,
  setVariablePartsAtStart,
  differentSeedForEach,
  setDifferentSeedForEach,
}) => {
  const handleImageSpacingChange = (value: number) => {
    setImageSpacing(value);
  };

  return (
    <>
      <div className="my-4">
        <label className="text-sm font-medium mb-2 block">b) Apply to:</label>
        <RadioGroup 
          value={promptTarget} 
          onValueChange={(value) => value && setPromptTarget(value)} 
          className="grid grid-cols-2 gap-3"
        >
          <div className="relative">
            <Card className={`flex items-center gap-3 p-3 cursor-pointer border ${promptTarget === "positive-prompt" ? "border-blue-600 bg-blue-50/50" : "border-gray-200"}`}>
              <RadioGroupItem id="positive-prompt" value="positive-prompt" className="h-4 w-4" />
              <label htmlFor="positive-prompt" className="text-sm font-medium cursor-pointer w-full">Positive Prompt</label>
            </Card>
          </div>
          
          <div className="relative">
            <Card className={`flex items-center gap-3 p-3 cursor-pointer border ${promptTarget === "negative-prompt" ? "border-blue-600 bg-blue-50/50" : "border-gray-200"}`}>
              <RadioGroupItem id="negative-prompt" value="negative-prompt" className="h-4 w-4" />
              <label htmlFor="negative-prompt" className="text-sm font-medium cursor-pointer w-full">Negative Prompt</label>
            </Card>
          </div>
        </RadioGroup>
        <div className="text-xs text-muted-foreground mt-1">Put the alternate prompts in {"{}"} brackets</div>
      </div>
      
      <div className="my-4">
        <label className="text-sm font-medium mb-2 block">c) Separate identifiers using:</label>
        <RadioGroup 
          value={separator} 
          onValueChange={(value) => value && setSeparator(value)} 
          className="grid grid-cols-2 gap-3"
        >
          <div className="relative">
            <Card className={`flex items-center gap-3 p-3 cursor-pointer border ${separator === "comma" ? "border-blue-600 bg-blue-50/50" : "border-gray-200"}`}>
              <RadioGroupItem id="comma" value="comma" className="h-4 w-4" />
              <label htmlFor="comma" className="text-sm font-medium cursor-pointer w-full">Comma</label>
            </Card>
          </div>
          
          <div className="relative">
            <Card className={`flex items-center gap-3 p-3 cursor-pointer border ${separator === "space" ? "border-blue-600 bg-blue-50/50" : "border-gray-200"}`}>
              <RadioGroupItem id="space" value="space" className="h-4 w-4" />
              <label htmlFor="space" className="text-sm font-medium cursor-pointer w-full">Space</label>
            </Card>
          </div>
        </RadioGroup>
        <div className="text-xs text-muted-foreground mt-1">Use letters / characters to build each picture</div>

        <div className="mt-4 mb-0">
          <div className="flex items-center justify-between">
            <div className="text-sm font-medium">
              <span>d) Image spacing (px):</span>
            </div>
            <div className="flex items-center text-xs">
              <span className="mr-1 text-muted-foreground">Min: 1</span>
              <span className="mx-1 text-muted-foreground">Max: 50</span>
              <input
                type="number"
                className="rounded-md border border-input bg-background px-2 py-1 text-right text-xs w-16"
                min="1"
                max="50"
                step="1"
                value={imageSpacing}
                onChange={(e) => {
                  const newValue = Number(e.target.value);
                  setImageSpacing(newValue);
                  handleImageSpacingChange(newValue);
                }}
              />
            </div>
          </div>
          <div className="py-0">
            <Slider
              min={1}
              max={50}
              step={1}
              defaultValue={4}
              label=""
              onChange={handleImageSpacingChange}
              hideInput={true}
            />
          </div>

          <div className="flex items-center gap-2 mt-3 mb-3">
            <Checkbox 
              id="variable-parts-start" 
              checked={variablePartsAtStart}
              onCheckedChange={(checked) => setVariablePartsAtStart(checked === true)}
            />
            <label htmlFor="variable-parts-start" className="text-sm">
              Put variable parts at start of prompt
            </label>
          </div>

          <div className="flex items-center gap-2 mb-3">
            <Checkbox 
              id="different-seed" 
              checked={differentSeedForEach}
              onCheckedChange={(checked) => setDifferentSeedForEach(checked === true)}
            />
            <label htmlFor="different-seed" className="text-sm">
              Use different seed for each picture
            </label>
          </div>
        </div>
      </div>
    </>
  );
};

export default PromptMatrixSettings;

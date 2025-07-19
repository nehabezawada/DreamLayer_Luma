
import React from "react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

interface PromptWeightsSettingsProps {
  promptTarget: string;
  setPromptTarget: (value: string) => void;
}

const PromptWeightsSettings: React.FC<PromptWeightsSettingsProps> = ({
  promptTarget,
  setPromptTarget,
}) => {
  return (
    <>
      <div className="my-4">
        <label className="text-sm font-medium mb-2 block">Apply to:</label>
        <RadioGroup 
          value={promptTarget} 
          onValueChange={(value) => value && setPromptTarget(value)} 
          className="grid grid-cols-2 gap-3"
        >
          <div className="relative">
            <Card className={`flex items-center gap-3 p-3 cursor-pointer border ${promptTarget === "positive-prompt" ? "border-blue-600 bg-blue-50/50" : "border-gray-200"}`}>
              <RadioGroupItem id="positive-prompt-weights" value="positive-prompt" className="h-4 w-4" />
              <label htmlFor="positive-prompt-weights" className="text-sm font-medium cursor-pointer w-full">Positive Prompt</label>
            </Card>
          </div>
          
          <div className="relative">
            <Card className={`flex items-center gap-3 p-3 cursor-pointer border ${promptTarget === "negative-prompt" ? "border-blue-600 bg-blue-50/50" : "border-gray-200"}`}>
              <RadioGroupItem id="negative-prompt-weights" value="negative-prompt" className="h-4 w-4" />
              <label htmlFor="negative-prompt-weights" className="text-sm font-medium cursor-pointer w-full">Negative Prompt</label>
            </Card>
          </div>
        </RadioGroup>
      </div>
      
      <div className="my-4">
        <label htmlFor="prompt-with-weights" className="text-sm font-medium mb-2 block">Prompt with weights:</label>
        <Textarea 
          id="prompt-with-weights" 
          className="mt-1" 
          placeholder="Enter prompt with weights in (text:1.5) format"
        />
        <div className="text-xs text-muted-foreground mt-1">Example: a (cat:1.2) chasing a (mouse:0.8)</div>
      </div>
      
      <div className="my-3">
        <label htmlFor="min-weight" className="text-sm font-medium">Min weight:</label>
        <Input
          id="min-weight"
          type="number"
          defaultValue="0.1"
          min="0"
          step="0.1"
          className="w-24 mt-1"
        />
      </div>
      
      <div className="my-3">
        <label htmlFor="max-weight" className="text-sm font-medium">Max weight:</label>
        <Input
          id="max-weight"
          type="number"
          defaultValue="2.0"
          min="0"
          step="0.1"
          className="w-24 mt-1"
        />
      </div>
      
      <div className="my-3">
        <label htmlFor="weight-step" className="text-sm font-medium">Weight step:</label>
        <Input
          id="weight-step"
          type="number"
          defaultValue="0.1"
          min="0.1"
          step="0.1"
          className="w-24 mt-1"
        />
      </div>
    </>
  );
};

export default PromptWeightsSettings;

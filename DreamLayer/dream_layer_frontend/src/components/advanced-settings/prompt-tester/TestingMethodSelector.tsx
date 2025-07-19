
import React from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface TestingMethodSelectorProps {
  testingMethod: string;
  onTestingMethodChange: (value: string) => void;
}

const TestingMethodSelector: React.FC<TestingMethodSelectorProps> = ({ 
  testingMethod,
  onTestingMethodChange
}) => {
  return (
    <div className="my-3">
      <label htmlFor="prompt-testing-method" className="text-sm font-medium">a) Script - Prompt Testing Method</label>
      <Select 
        defaultValue="none" 
        value={testingMethod} 
        onValueChange={onTestingMethodChange}
      >
        <SelectTrigger className="w-full mt-1" id="prompt-testing-method">
          <SelectValue placeholder="Select a testing method" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="none">None</SelectItem>
          <SelectItem value="prompt-matrix">Prompt Matrix</SelectItem>
          <SelectItem value="x-y-plot">X/Y/Z Plot</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
};

export default TestingMethodSelector;

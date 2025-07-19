
import React, { useState } from "react";

// Import sub-components
import TestingMethodSelector from "./prompt-tester/TestingMethodSelector";
import PromptMatrixSettings from "./prompt-tester/PromptMatrixSettings";
import XYPlotSettings from "./prompt-tester/XYPlotSettings";

const PromptTesterSettings: React.FC = () => {
  const [testingMethod, setTestingMethod] = useState("none");
  const [promptTarget, setPromptTarget] = useState("positive-prompt");
  const [separator, setSeparator] = useState("comma");
  const [imageSpacing, setImageSpacing] = useState(4);
  const [variablePartsAtStart, setVariablePartsAtStart] = useState(false);
  const [differentSeedForEach, setDifferentSeedForEach] = useState(false);
  
  const handleTestingMethodChange = (value: string) => {
    setTestingMethod(value);
  };

  return (
    <>
      <TestingMethodSelector 
        testingMethod={testingMethod} 
        onTestingMethodChange={handleTestingMethodChange} 
      />
      
      {testingMethod === "prompt-matrix" && (
        <PromptMatrixSettings 
          promptTarget={promptTarget}
          setPromptTarget={setPromptTarget}
          separator={separator}
          setSeparator={setSeparator}
          imageSpacing={imageSpacing}
          setImageSpacing={setImageSpacing}
          variablePartsAtStart={variablePartsAtStart}
          setVariablePartsAtStart={setVariablePartsAtStart}
          differentSeedForEach={differentSeedForEach}
          setDifferentSeedForEach={setDifferentSeedForEach}
        />
      )}
      
      {testingMethod === "x-y-plot" && (
        <XYPlotSettings />
      )}
    </>
  );
};

export default PromptTesterSettings;

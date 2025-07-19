
import React, { useState, useEffect } from 'react';
import Img2ImgHeader from './img2img/Img2ImgHeader';
import SettingsAccordion from './img2img/SettingsAccordion';
import ImagePreview from './img2img/ImagePreview';

const Img2ImgTab = () => {
  const [activeSubTab, setActiveSubTab] = useState("generation");
  const [batchCount, setBatchCount] = useState(25);
  const [batchSize, setBatchSize] = useState(4);
  const [isLoaded, setIsLoaded] = useState(false);
  
  useEffect(() => {
    // Mark component as loaded after mounting
    setIsLoaded(true);
    console.log("Img2ImgTab component mounted");
  }, []);
  
  const handleSubTabChange = (tabId: string) => {
    setActiveSubTab(tabId);
  };
  
  // Render a simple debug message if we're in a loading state
  if (!isLoaded) {
    return (
      <div className="p-8 bg-card text-foreground border border-border rounded-md">
        <p>Loading Img2Img interface...</p>
      </div>
    );
  }

  return (
    <div className="mb-4 grid gap-6 md:grid-cols-[2fr_1fr] bg-background">
      {/* Left Column - Controls */}
      <div className="space-y-4">
        <div className="flex flex-col">
          <Img2ImgHeader 
            activeSubTab={activeSubTab} 
            onTabChange={handleSubTabChange} 
          />
          
          <SettingsAccordion 
            activeSubTab={activeSubTab}
            batchCount={batchCount}
            setBatchCount={setBatchCount}
            batchSize={batchSize}
            setBatchSize={setBatchSize}
          />
        </div>
      </div>
      
      {/* Right Column - Preview */}
      <div>
        <ImagePreview />
      </div>
    </div>
  );
};

export default Img2ImgTab;

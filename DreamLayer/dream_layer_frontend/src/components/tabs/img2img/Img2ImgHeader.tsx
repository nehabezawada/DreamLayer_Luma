import React from 'react';
import SubTabNavigation from '@/components/SubTabNavigation';

interface Img2ImgHeaderProps {
  activeSubTab: string;
  onTabChange: (tabId: string) => void;
}

const Img2ImgHeader: React.FC<Img2ImgHeaderProps> = ({ activeSubTab, onTabChange }) => {
  const generationTabs = [
    { id: "generation", label: "Generation", active: activeSubTab === "generation" },
    { id: "checkpoints", label: "Custom Workflow", active: activeSubTab === "checkpoints" },
    { id: "lora", label: "Lora", active: activeSubTab === "lora" }
  ];

  return (
    <>
      <div className="mb-[18px] flex flex-col space-y-2 sm:flex-row sm:items-center sm:justify-between sm:space-y-0">
        <h3 className="text-base font-medium text-foreground">Image to Image Generation</h3>
        <div className="flex space-x-2">
          <button className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90">
            Generate Image
          </button>
          <button className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground">
            Save Settings
          </button>
        </div>
      </div>
      
      <SubTabNavigation 
        className="mb-[18px]"
        tabs={generationTabs}
        onTabChange={onTabChange}
      />
    </>
  );
};

export default Img2ImgHeader;

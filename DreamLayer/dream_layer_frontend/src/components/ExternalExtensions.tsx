import React from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { ChevronUp } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import ControlNetInterface from "./controlnet/ControlNetInterface";
import { ControlNetRequest } from "@/types/controlnet";
import useControlNetStore from "@/stores/useControlNetStore";

const extensions = [
  { id: "controlnet", label: "ControlNet v1.1.447+" },
];

interface ExternalExtensionsProps {
  isImg2ImgTab?: boolean;
  onControlNetChange?: (controlNet: ControlNetRequest) => void;
}

const ExternalExtensions: React.FC<ExternalExtensionsProps> = ({ 
  isImg2ImgTab = false,
  onControlNetChange 
}) => {
  const {
    isControlNetEnabled,
    setIsControlNetEnabled,
    isControlNetOpen,
    setIsControlNetOpen
  } = useControlNetStore();

  const toggleExtension = (id: string) => {
    if (id === 'controlnet') {
      const newEnabled = !isControlNetEnabled;
      console.log(`[ControlNet] Checkbox clicked. New enabled state:`, newEnabled);
      setIsControlNetEnabled(newEnabled);

      // Auto-open the panel when enabling
      if (newEnabled && !isControlNetOpen) {
        setIsControlNetOpen(true);
        console.log('[ControlNet] Auto-opening panel because ControlNet was enabled');
      }

      // If disabling ControlNet, clear the configuration
      if (!newEnabled && onControlNetChange) {
        onControlNetChange(null);
      }
    }
  };

  const toggleOpen = (id: string) => {
    if (id === 'controlnet') {
      const newOpen = !isControlNetOpen;
      console.log(`[ControlNet] Panel toggled. New open state:`, newOpen);
      setIsControlNetOpen(newOpen);
    }
  };

  console.log(`[ControlNet] Render: isControlNetEnabled=`, isControlNetEnabled, 'isControlNetOpen=', isControlNetOpen);

  return (
    <div className="space-y-3">
      {extensions.map((extension) => (
        <Collapsible
          key={extension.id}
          open={extension.id === 'controlnet' ? isControlNetOpen : false}
          onOpenChange={() => toggleOpen(extension.id)}
          className="w-full"
        >
          <div className="flex items-center border border-border rounded-md">
            <div className="flex items-center justify-between w-full p-3">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id={extension.id}
                  checked={extension.id === 'controlnet' ? isControlNetEnabled : false}
                  onCheckedChange={() => toggleExtension(extension.id)}
                  className="h-4 w-4 text-primary"
                />
                <label
                  htmlFor={extension.id}
                  className="text-sm font-medium text-blue-500"
                >
                  {extension.label}
                </label>
              </div>
              <CollapsibleTrigger asChild>
                <ChevronUp
                  className={`h-4 w-4 text-muted-foreground transition-transform ${
                    (extension.id === 'controlnet' ? isControlNetOpen : false) ? "" : "transform rotate-180"
                  }`}
                />
              </CollapsibleTrigger>
            </div>
          </div>
          <CollapsibleContent className="overflow-hidden border-x border-b border-border rounded-b-md mt-[-1px]">
            {/* Only render the interface if the extension is enabled */}
            {(() => {
              if (extension.id === 'controlnet' ? isControlNetEnabled : false) {
                console.log('[ControlNet] Rendering ControlNetInterface');
                return (
                  <ControlNetInterface
                    isImg2ImgTab={isImg2ImgTab}
                    onControlNetChange={onControlNetChange}
                  />
                );
              }
              return null;
            })()}
          </CollapsibleContent>
        </Collapsible>
      ))}
    </div>
  );
};

export default ExternalExtensions;

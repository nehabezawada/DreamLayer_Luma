import React from "react";
import { Slider } from "@/components/ui/slider";

interface TilingSettingsProps {
  tileSize?: number;
  setTileSize?: (size: number) => void;
  overlap?: number;
  setOverlap?: (overlap: number) => void;
}

const TilingSettings: React.FC<TilingSettingsProps> = ({
  tileSize = 512,
  setTileSize,
  overlap = 64,
  setOverlap
}) => {
  const handleTileSizeChange = (value: number[]) => {
    setTileSize?.(value[0]);
  };

  const handleOverlapChange = (value: number[]) => {
    setOverlap?.(value[0]);
  };

  return (
    <div className="space-y-4">
      <div className="text-sm text-muted-foreground mb-3">
        Tiling allows you to generate seamless patterns by processing the image in tiles. 
        This is useful for creating repeating textures and patterns.
      </div>
      
      <div className="space-y-4">
        <div>
          <label className="text-sm font-medium mb-2 block">
            Tile Size: {tileSize}px
          </label>
          <Slider
            min={64}
            max={1024}
            step={32}
            defaultValue={[tileSize]}
            onValueChange={handleTileSizeChange}
            className="w-full"
          />
          <div className="text-xs text-muted-foreground mt-1">
            Size of each tile. Larger tiles use more memory but may produce better results.
          </div>
        </div>
        
        <div>
          <label className="text-sm font-medium mb-2 block">
            Overlap: {overlap}px
          </label>
          <Slider
            min={0}
            max={256}
            step={32}
            defaultValue={[overlap]}
            onValueChange={handleOverlapChange}
            className="w-full"
          />
          <div className="text-xs text-muted-foreground mt-1">
            Overlap between tiles. Higher overlap reduces seams but uses more memory.
          </div>
        </div>
      </div>
      
      <div className="bg-blue-50 dark:bg-blue-950/20 p-3 rounded-md">
        <div className="text-xs text-blue-700 dark:text-blue-300">
          <strong>Tip:</strong> For seamless patterns, try tile sizes of 512-768px with 64-128px overlap.
        </div>
      </div>
    </div>
  );
};

export default TilingSettings; 
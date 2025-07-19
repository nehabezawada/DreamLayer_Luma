
import React, { useState } from "react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Search } from "lucide-react";

const styleOptions = [
  "3D Model", "Abstract", "Advertising", "Alien", "Analog Film", "Anime", "Architectural",
  "Cinematic", "Collage", "Comic Book", "Cubist", "Digital Art", "Disco", "Dreamscape",
  "Dystopian", "Enhance", "Fairy Tale", "Fantasy Art", "Fighting Game", "Film Noir",
  "Flat Papercut", "Food Photography", "GTA", "Gothic", "Graffiti", "Grunge", "HDR",
  "Horror", "Hyperrealism", "Impressionist", "Isometric Style", "Kirigami", "Legend of Zelda",
  "Line Art", "Long Exposure", "Lowpoly", "Minecraft", "Minimalist", "Monochrome",
  "Nautical", "Neon Noir", "Neon Punk", "Origami", "Paper Mache", "Paper Quilling", 
  "Papercut Collage", "Papercut Shadow Box", "Photographic", "Pixel Art", "Pointillism",
  "Pokémon", "Pop Art", "Psychedelic", "RPG Fantasy Game", "Real Estate", "Renaissance",
  "Retro Arcade", "Retro Game", "Silhouette", "Space", "Stacked Papercut", "Stained Glass", 
  "Steampunk", "Strategy Game", "Street Fighter", "Super Mario", "Surrealist", 
  "Techwear Fashion", "Texture", "Thick Layered Papercut", "Tilt-Shift", "Tribal", 
  "Typography", "Watercolor", "Zentangle", "base"
];

const SDXLStyleSettings: React.FC = () => {
  const [styleMode, setStyleMode] = useState<string>("random");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedStyle, setSelectedStyle] = useState<string>("");
  
  // Filter styles based on search query
  const filteredStyles = styleOptions.filter(style => 
    style.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Check if the current mode is manual
  const isManualMode = styleMode === "manual";

  return (
    <div className="my-4">
      <p className="text-sm font-medium mb-2">How should styles be applied?</p>
      
      <RadioGroup 
        value={styleMode}
        onValueChange={setStyleMode}
        className="space-y-3"
      >
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="random" id="style-random" />
          <Label htmlFor="style-random">Randomize Style – One style for the whole generation</Label>
        </div>
        
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="each-image" id="style-each" />
          <Label htmlFor="style-each">Randomize style for each image of the generation</Label>
        </div>
        
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="all" id="style-all" />
          <Label htmlFor="style-all">Generate all styles in order (Output Quantity: 77)</Label>
        </div>
        
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="manual" id="style-manual" />
          <Label htmlFor="style-manual">Pick style manually</Label>
        </div>
      </RadioGroup>
      
      {/* Always show the style selector but conditionally apply disabled state */}
      <div className={`mt-4 border border-border rounded-md p-3 ${!isManualMode ? 'opacity-60' : ''}`}>
        <div className="relative">
          <Label htmlFor="search-styles" className="mb-2 block">All Styles:</Label>
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              id="search-styles"
              placeholder="Search Styles"
              className="pl-9"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              disabled={!isManualMode}
            />
          </div>
        </div>
        
        <ScrollArea className="h-[280px] mt-3 pr-4">
          <RadioGroup 
            value={selectedStyle} 
            onValueChange={setSelectedStyle}
            className="grid grid-cols-3 gap-2"
            disabled={!isManualMode}
          >
            {filteredStyles.map((style) => (
              <div
                key={style}
                className={`
                  flex items-center gap-2 p-2 rounded-full border 
                  ${selectedStyle === style ? 'border-primary bg-primary/10' : 'border-border'}
                  ${isManualMode ? 'hover:border-primary/50' : ''} 
                  transition-colors cursor-pointer
                  ${!isManualMode ? 'pointer-events-none' : ''}
                `}
                onClick={() => isManualMode && setSelectedStyle(style)}
              >
                <RadioGroupItem 
                  value={style} 
                  id={`style-${style}`} 
                  className="data-[state=checked]:border-primary"
                  disabled={!isManualMode}
                />
                <Label 
                  htmlFor={`style-${style}`} 
                  className={`text-xs cursor-pointer truncate ${!isManualMode ? 'cursor-default' : ''}`}
                >
                  {style}
                </Label>
              </div>
            ))}
          </RadioGroup>
        </ScrollArea>
      </div>
    </div>
  );
};

export default SDXLStyleSettings;

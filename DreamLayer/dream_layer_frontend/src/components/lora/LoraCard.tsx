import React from 'react';
import { Button } from '@/components/ui/button';
import { LoraModel } from '@/types/lora';
import useLoraStore from '@/stores/useLoraStore';

interface LoraCardProps {
  model: LoraModel;
  viewMode: 'grid' | 'list';
}

const LoraCard = ({ model, viewMode }: LoraCardProps) => {
  const { loraConfig, setSelectedLora } = useLoraStore();
  const isSelected = loraConfig?.enabled && loraConfig?.lora_name === model.filename;

  const handleSelect = () => {
    setSelectedLora(isSelected ? null : model.filename, model.defaultWeight);
  };

  if (viewMode === 'list') {
    return (
      <div className={`flex items-center gap-3 p-3 border border-border rounded-md hover:bg-accent/50 transition-colors ${isSelected ? 'bg-accent' : ''}`}>
        <img
          src={model.preview || '/placeholder.svg'}
          alt={model.name}
          className="w-12 h-12 object-cover rounded bg-muted flex-shrink-0"
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div className="min-w-0 flex-1">
              <h4 className="font-medium text-sm truncate">{model.name}</h4>
            </div>
            <div className="flex items-center gap-2 ml-3">
              <Button
                size="sm"
                onClick={handleSelect}
                variant={isSelected ? "secondary" : "outline"}
                className="text-xs h-6 px-2"
              >
                {isSelected ? 'Selected' : 'Select'}
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`border border-border rounded-md overflow-hidden hover:bg-accent/50 transition-colors group ${isSelected ? 'bg-accent' : ''}`}>
      <div className="aspect-square relative">
        <img
          src={model.preview || '/placeholder.svg'}
          alt={model.name}
          className="w-full h-full object-cover bg-muted"
        />
        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
          <Button
            size="sm"
            onClick={handleSelect}
            variant={isSelected ? "secondary" : "outline"}
            className="text-xs"
          >
            {isSelected ? 'Selected' : 'Select'}
          </Button>
        </div>
      </div>
      <div className="p-2">
        <h4 className="font-medium text-xs truncate" title={model.name}>
          {model.name}
        </h4>
      </div>
    </div>
  );
};

export default LoraCard;

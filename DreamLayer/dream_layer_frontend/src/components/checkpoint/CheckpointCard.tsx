
import React from 'react';
import { Button } from '@/components/ui/button';

interface CheckpointModel {
  id: string;
  name: string;
  filename: string;
  preview?: string;
  description: string;
  tags: string[];
  version: string;
  baseModel: string;
  vae?: string;
}

interface CheckpointCardProps {
  model: CheckpointModel;
  viewMode: 'grid' | 'list';
  onSelect: (model: CheckpointModel) => void;
}

const CheckpointCard = ({ model, viewMode, onSelect }: CheckpointCardProps) => {
  if (viewMode === 'list') {
    return (
      <div className="flex items-center gap-3 p-3 border border-border rounded-md hover:bg-accent/50 transition-colors">
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
                onClick={() => onSelect(model)}
                className="text-xs h-6 px-2"
              >
                Select
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="border border-border rounded-md overflow-hidden hover:bg-accent/50 transition-colors group">
      <div className="aspect-square relative">
        <img
          src={model.preview || '/placeholder.svg'}
          alt={model.name}
          className="w-full h-full object-cover bg-muted"
        />
        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
          <Button
            size="sm"
            onClick={() => onSelect(model)}
            className="text-xs"
          >
            Select Model
          </Button>
        </div>
      </div>
      <div className="p-3">
        <h4 className="font-medium text-sm truncate" title={model.name}>
          {model.name}
        </h4>
      </div>
    </div>
  );
};

export default CheckpointCard;

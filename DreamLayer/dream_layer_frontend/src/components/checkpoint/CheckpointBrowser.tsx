
import React, { useState } from 'react';
import { Search, Grid, List, ArrowUp, ArrowDown, RefreshCw } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import CheckpointCard from './CheckpointCard';

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

const mockCheckpoints: CheckpointModel[] = [
  {
    id: 'sd15',
    name: 'Stable Diffusion 1.5',
    filename: 'v1-5-pruned-emaonly.safetensors',
    preview: '/placeholder.svg',
    description: 'The classic Stable Diffusion 1.5 base model',
    tags: ['base', 'stable', 'general'],
    version: '1.5',
    baseModel: 'SD 1.5',
    vae: 'SD 1.5 VAE'
  },
  {
    id: 'dreamshaper',
    name: 'DreamShaper 8',
    filename: 'dreamshaper_8.safetensors',
    preview: '/placeholder.svg',
    description: 'Popular checkpoint for fantasy and artistic images',
    tags: ['artistic', 'fantasy', 'dreamlike'],
    version: '8.0',
    baseModel: 'SD 1.5',
    vae: 'EMA VAE'
  },
  {
    id: 'realistic-vision',
    name: 'Realistic Vision 5.1',
    filename: 'realisticVisionV51_v51VAE.safetensors',
    preview: '/placeholder.svg',
    description: 'Highly realistic photo-style generations',
    tags: ['realistic', 'photo', 'portrait'],
    version: '5.1',
    baseModel: 'SD 1.5',
    vae: 'Built-in'
  },
  {
    id: 'sdxl-base',
    name: 'SDXL Base 1.0',
    filename: 'sd_xl_base_1.0.safetensors',
    preview: '/placeholder.svg',
    description: 'Stable Diffusion XL base model with improved quality',
    tags: ['sdxl', 'base', 'high-res'],
    version: '1.0',
    baseModel: 'SDXL',
    vae: 'SDXL VAE'
  },
  {
    id: 'anime-xl',
    name: 'Anime XL',
    filename: 'animeArtDiffusionXL_alpha3.safetensors',
    preview: '/placeholder.svg',
    description: 'SDXL fine-tuned for anime and manga style art',
    tags: ['anime', 'manga', 'sdxl', 'style'],
    version: 'Alpha 3',
    baseModel: 'SDXL',
    vae: 'SDXL VAE'
  },
  {
    id: 'deliberate',
    name: 'Deliberate v2',
    filename: 'deliberate_v2.safetensors',
    preview: '/placeholder.svg',
    description: 'Versatile model good for various art styles',
    tags: ['versatile', 'artistic', 'balanced'],
    version: '2.0',
    baseModel: 'SD 1.5',
    vae: 'MSE VAE'
  }
];

const CheckpointBrowser = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [sortBy, setSortBy] = useState('name');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');

  const filteredModels = mockCheckpoints.filter(model => {
    const matchesSearch = model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         model.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         model.description.toLowerCase().includes(searchTerm.toLowerCase());
    
    return matchesSearch;
  });

  const sortedModels = [...filteredModels].sort((a, b) => {
    let comparison = 0;
    switch (sortBy) {
      case 'name':
        comparison = a.name.localeCompare(b.name);
        break;
      case 'baseModel':
        comparison = a.baseModel.localeCompare(b.baseModel);
        break;
      case 'filename':
        comparison = a.filename.localeCompare(b.filename);
        break;
      default:
        return 0;
    }
    return sortOrder === 'desc' ? -comparison : comparison;
  });

  const handleCheckpointSelect = (model: CheckpointModel) => {
    console.log(`Selected checkpoint: ${model.name}`);
    // In a real implementation, this would update the selected checkpoint
  };

  const toggleSortOrder = () => {
    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
  };

  const handleRefresh = () => {
    console.log('Refreshing checkpoint list...');
    // In a real implementation, this would refetch the checkpoint data
  };

  return (
    <div className="space-y-4">
      {/* Search and Controls */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <input
              type="text"
              placeholder="Search checkpoints..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-input rounded-md bg-background text-sm"
            />
          </div>
          <Select value={sortBy} onValueChange={setSortBy}>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="Sort by..." />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="name">Sort by Name</SelectItem>
              <SelectItem value="baseModel">Sort by Base Model</SelectItem>
              <SelectItem value="filename">Sort by Filename</SelectItem>
            </SelectContent>
          </Select>
          <button
            onClick={toggleSortOrder}
            className="p-2 border border-input rounded-md hover:bg-accent transition-colors"
          >
            {sortOrder === 'asc' ? (
              <ArrowUp className="h-4 w-4" />
            ) : (
              <ArrowDown className="h-4 w-4" />
            )}
          </button>
          <button
            onClick={handleRefresh}
            className="p-2 border border-input rounded-md hover:bg-accent transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
          <div className="flex border border-input rounded-md">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 ${viewMode === 'grid' ? 'bg-accent' : ''}`}
            >
              <Grid className="h-4 w-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 ${viewMode === 'list' ? 'bg-accent' : ''}`}
            >
              <List className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Results count */}
      <div className="text-sm text-muted-foreground">
        {sortedModels.length} checkpoint{sortedModels.length !== 1 ? 's' : ''} found
      </div>

      {/* Models grid/list */}
      <div className={viewMode === 'grid' 
        ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3" 
        : "space-y-2"
      }>
        {sortedModels.map(model => (
          <CheckpointCard
            key={model.id}
            model={model}
            viewMode={viewMode}
            onSelect={handleCheckpointSelect}
          />
        ))}
      </div>

      {sortedModels.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <div className="text-4xl mb-2">🔍</div>
          <div>No checkpoints found</div>
          <div className="text-sm">Try adjusting your search or filters</div>
        </div>
      )}
    </div>
  );
};

export default CheckpointBrowser;

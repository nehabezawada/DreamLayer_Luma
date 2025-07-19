import React, { useState } from "react";
import { Button } from "@/components/ui/button";

interface ControlNetImage {
  url: string;  // For preview
  file: File;   // The actual file
  filename?: string; // Server filename after upload
}

interface ControlNetImageUploadProps {
  onImageChange?: (image: ControlNetImage | null) => void;
  unitIndex?: number; // Add unit index for upload
}

const ControlNetImageUpload: React.FC<ControlNetImageUploadProps> = ({ onImageChange, unitIndex = 0 }) => {
  const [imageData, setImageData] = useState<ControlNetImage | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const uploadImageToServer = async (file: File): Promise<string | null> => {
    try {
      console.log('ControlNetImageUpload: Uploading image to server...');
      
      const formData = new FormData();
      formData.append('file', file);
      formData.append('unit_index', unitIndex.toString());
      
      const response = await fetch('http://localhost:5001/api/upload-controlnet-image', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('ControlNetImageUpload: Upload failed:', errorText);
        throw new Error(`Upload failed: ${errorText}`);
      }
      
      const result = await response.json();
      console.log('ControlNetImageUpload: Upload successful:', result);
      
      if (result.status === 'success') {
        return result.filename;
      } else {
        throw new Error(result.message || 'Upload failed');
      }
      
    } catch (error) {
      console.error('ControlNetImageUpload: Error uploading image:', error);
      throw error;
    }
  };

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    console.log('ControlNetImageUpload: File upload triggered');
    const file = event.target.files?.[0];
    console.log('ControlNetImageUpload: File selected:', file);
    
    if (file) {
      console.log('ControlNetImageUpload: File details:', {
        name: file.name,
        size: file.size,
        type: file.type
      });
      
      setIsUploading(true);
      
      try {
        // Upload to server first
        const filename = await uploadImageToServer(file);
        
        if (filename) {
          // Create URL for preview (use server URL)
          const url = `http://localhost:5001/api/images/${filename}`;
          
          // Store the image data with server filename
          const newImageData = { url, file, filename };
          setImageData(newImageData);
          
          console.log('ControlNetImageUpload: Calling onImageChange with:', newImageData);
          // Notify parent
          onImageChange?.(newImageData);
        }
      } catch (error) {
        console.error('ControlNetImageUpload: Failed to upload image:', error);
        // Fallback to local preview if upload fails
        const url = URL.createObjectURL(file);
        const newImageData = { url, file };
        setImageData(newImageData);
        onImageChange?.(newImageData);
      } finally {
        setIsUploading(false);
      }
    } else {
      console.log('ControlNetImageUpload: No file selected');
    }
  };

  const handleRunPreview = () => {
    // TODO: Implement preview functionality
  };

  const handleClearImage = () => {
    if (imageData?.url && !imageData.filename) {
      // Only revoke URL if it's a local blob URL (not a server URL)
      URL.revokeObjectURL(imageData.url);
    }
    setImageData(null);
    onImageChange?.(null);
  };

  return (
    <div className="w-full space-y-2">
      {/* Header with Image label and Run Preview button */}
      <div className="flex items-center justify-between">
        <div className="font-medium text-sm text-foreground">Image</div>
        <Button 
          variant="outline" 
          size="sm" 
          className="text-xs"
          onClick={handleRunPreview}
          disabled={isUploading}
        >
          {isUploading ? 'Uploading...' : 'Run Preview'}
        </Button>
      </div>
      
      {imageData ? (
        <div className="relative">
          <img 
            src={imageData.url} 
            alt="ControlNet input" 
            className="rounded-md object-cover w-full h-full border border-border"
          />
          <div className="absolute bottom-2 right-2 flex gap-1">
            <Button 
              variant="outline" 
              size="sm" 
              className="bg-background/80 backdrop-blur-sm border-border text-foreground hover:bg-accent"
              onClick={handleClearImage}
            >
              Clear
            </Button>
          </div>
          {imageData.filename && (
            <div className="absolute top-2 left-2 bg-green-500 text-white text-xs px-2 py-1 rounded">
              ✓ Uploaded
            </div>
          )}
        </div>
      ) : (
        <div className="flex items-center justify-center w-full">
          <label className={`flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-background hover:bg-accent/10 ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}>
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              {isUploading ? (
                <div className="flex flex-col items-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
                  <p className="text-sm text-muted-foreground">Uploading...</p>
                </div>
              ) : (
                <>
                  <svg className="w-8 h-8 mb-4 text-muted-foreground" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                    <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
                  </svg>
                  <p className="mb-2 text-sm text-muted-foreground">
                    <span className="font-semibold">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-xs text-muted-foreground">PNG, JPG or JPEG</p>
                </>
              )}
            </div>
            <input 
              type="file" 
              className="hidden" 
              accept="image/png,image/jpeg,image/jpg"
              onChange={handleImageUpload}
              disabled={isUploading}
            />
          </label>
        </div>
      )}
    </div>
  );
};

export default ControlNetImageUpload;

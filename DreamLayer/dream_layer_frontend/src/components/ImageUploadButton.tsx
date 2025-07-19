import React, { useRef } from 'react';
import { Button } from "@/components/ui/button";

interface ImageUploadButtonProps {
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  accept?: string;
  className?: string;
  children: React.ReactNode;
}

const ImageUploadButton: React.FC<ImageUploadButtonProps> = ({ 
  onFileChange, 
  accept = "image/*", 
  className = "",
  children 
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="inline-block">
      <Button 
        variant="secondary" 
        size="sm" 
        className={className}
        type="button"
        onClick={handleClick}
      >
        {children}
      </Button>
      <input 
        ref={fileInputRef}
        type="file" 
        accept={accept}
        className="hidden" 
        onChange={onFileChange}
      />
    </div>
  );
};

export default ImageUploadButton;

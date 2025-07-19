
import { AspectRatio } from "@/components/ui/aspect-ratio";
import { useTxt2ImgGalleryStore } from '@/stores/useTxt2ImgGalleryStore';

const ImagePreview = () => {
  const { images } = useTxt2ImgGalleryStore();

  return (
    <AspectRatio ratio={1} className="w-full">
      <div className="flex h-full w-full flex-col items-center justify-center rounded-md border border-border bg-card p-4">
        {images.length > 0 ? (
          <div className="w-full h-full">
            <img 
              src={images[0].url} 
              alt="Generated image" 
              className="w-full h-full object-cover rounded-md"
            />
          </div>
        ) : (
          <>
            <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-md bg-secondary">
              <svg
                viewBox="0 0 24 24"
                className="h-6 w-6 text-muted-foreground"
                fill="none"
                stroke="currentColor"
                xmlns="http://www.w3.org/2000/svg"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <path d="m21 15-5-5L5 21" />
              </svg>
            </div>
            <p className="text-sm text-muted-foreground">Generated Images Will Display Here</p>
          </>
        )}
      </div>
    </AspectRatio>
  );
};

export default ImagePreview;

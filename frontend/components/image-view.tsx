'use client';

import React, { useState, useEffect } from 'react';
import { Maximize } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import Masonry from '@/components/masonry-layout';

interface ImageLightboxProps {
  isOpen: boolean;
  onClose: () => void;
  images: string[];
  currentIndex: number;
  setCurrentIndex: (index: number) => void;
}

const ImageLightbox = ({ isOpen, onClose, images, currentIndex, setCurrentIndex }: ImageLightboxProps) => (
  <Dialog open={isOpen} onOpenChange={onClose}>
    <DialogContent className="max-w-4xl">
      <DialogHeader>
        <DialogTitle>Image Viewer</DialogTitle>
      </DialogHeader>
      <div className="flex items-center justify-center">
        <img 
          src={images[currentIndex]} 
          alt={`Image ${currentIndex + 1}`} 
          className="max-h-[70vh] w-auto object-contain"
        />
      </div>
      {images.length > 1 && (
        <div className="flex justify-center space-x-4 mt-4">
          {images.map((_, index) => (
            <Button 
              key={index} 
              variant={currentIndex === index ? 'default' : 'outline'}
              onClick={() => setCurrentIndex(index)}
            >
              {index + 1}
            </Button>
          ))}
        </div>
      )}
    </DialogContent>
  </Dialog>
);

interface ImageViewProps {
  image: string;
  album?: string[];
  title?: string;
  source?: string;
  similar: string[];
}

const ImageView: React.FC<ImageViewProps> = ({
  image,
  album = [],
  title = 'No title available',
  source = 'No source available',
  similar
}) => {
  const [isLightboxOpen, setIsLightboxOpen] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  const images = album.length ? album : [image];
  const placeholderImages = Array(6).fill('/api/placeholder/100/100');

  // aleays start the component from top
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <>
    <div className="w-full h-full flex flex-col overflow-hidden">
      <div className="flex flex-col md:flex-row flex-1 overflow-hidden relative">
        <div className="md:w-[76%] flex items-center justify-center p-4">
          <div className="relative group">
            <img 
              src={image} 
              alt="Detailed View" 
              className="max-h-[calc(100vh-200px)] w-auto object-contain cursor-pointer"
              onClick={() => setIsLightboxOpen(true)}
            />
            <Button 
              variant="ghost" 
              size="icon" 
              className="absolute top-2 right-2 opacity-100 transition-opacity"
              onClick={() => setIsLightboxOpen(true)}
            >
              <Maximize className="h-5 w-5" />
            </Button>
          </div>
        </div>

        <Separator orientation="vertical" className="hidden md:block" />

        <div className="mx-4 md:w-[24%] p-4 overflow-auto text-black dark:text-white lg:mr-6">
          <h2 className="text-xl font-bold mb-4">{title}</h2>
          <div className="bg-secondary p-3 rounded-md mb-4">
            <p className="text-sm">{source}</p>
          </div>
          <h3 className="text-lg font-semibold mb-3">Related Images</h3>
          <div className="grid grid-cols-3 gap-2">
            {placeholderImages.map((src, index) => (
              <img 
                key={index}
                src={src}
                alt={`Placeholder ${index + 1}`}
                className="w-full h-auto rounded-md"
              />
            ))}
          </div>
        </div>
      </div>

      <ImageLightbox 
        isOpen={isLightboxOpen}
        onClose={() => setIsLightboxOpen(false)}
        images={images}
        currentIndex={currentImageIndex}
        setCurrentIndex={setCurrentImageIndex}
      />
    </div>
    {/* Add Title similar images */}
            <Separator orientation="horizontal" className="hidden md:block" />
    
    <h2 className="text-2xl font-semibold text-black dark:text-white my-6 text-center">Similar Images</h2>    
    <Masonry initialImages={similar} />
    </>
  );
};

export default ImageView;
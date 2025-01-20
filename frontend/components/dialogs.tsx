import React, { useState, useRef } from 'react';
import { Dialog, DialogHeader, DialogTitle, DialogContent } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Loader2 } from 'lucide-react';
import Image from 'next/image';
import { toast } from 'sonner';
import { createCluster, createElement } from '@/lib/api';

interface DialogProps {
  onClose: () => void;
}

interface ImageFormState {
  file: File | null;
  preview: string;
  title: string;
  desc: string;
  cluster: string;
  loading: boolean;
}

const initialImageState: ImageFormState = {
  file: null,
  preview: '',
  title: '',
  desc: '',
  cluster: '',
  loading: false,
};

export const ImageDialog: React.FC<DialogProps> = ({ onClose }) => {
  const [state, setState] = useState<ImageFormState>(initialImageState);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file?.type.startsWith('image/')) {
      setState(prev => ({
        ...prev,
        file,
        preview: URL.createObjectURL(file)
      }));
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setState(prev => ({
        ...prev,
        file,
        preview: URL.createObjectURL(file)
      }));
    }
  };

  const handleSubmit = async () => {
    if (!state.file) {
      toast.error('Please select an image');
      return;
    }

    setState(prev => ({ ...prev, loading: true }));
    try {
      const formData = new FormData();
      formData.append('title', state.title);
      formData.append('desc', state.desc);
      formData.append('cluster', state.cluster);
      formData.append('image', state.file);
      
      const { error } = await createElement(formData);
      if (error) throw error;
      toast.success('Image uploaded successfully');
      setState(initialImageState);
      onClose();
    } catch (error) {
      console.error('Upload failed:', error);
      toast.error('Upload failed');
    } finally {
      setState(prev => ({ ...prev, loading: false }));
    }
  };

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Upload Image</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="border-2 border-dashed rounded-lg p-4 text-center hover:border-blue-500 transition-colors"
          >
            {state.file ? (
              <div className="relative h-48 w-full">
                <Image
                  src={state.preview}
                  alt="Preview"
                  fill
                  className="object-contain rounded"
                />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center space-y-3 py-8">
                <Image src="/assets/upload.png" alt="upload" width={40} height={40} />
                <p className="text-sm font-medium text-gray-600">
                  Drag & drop or click to upload
                </p>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="image-input"
                />
                <Button onClick={() => document.getElementById('image-input')?.click()} size="sm">
                  Choose File
                </Button>
              </div>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="title">Title</Label>
            <Input
              id="title"
              value={state.title}
              onChange={e => setState(prev => ({ ...prev, title: e.target.value }))}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="desc">Description</Label>
            <Textarea
              id="desc"
              className="resize-none"
              value={state.desc}
              onChange={e => setState(prev => ({ ...prev, desc: e.target.value }))}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="cluster">Cluster</Label>
            <Input
              id="cluster"
              value={state.cluster}
              onChange={e => setState(prev => ({ ...prev, cluster: e.target.value }))}
            />
          </div>

          <div className="flex justify-end space-x-2 pt-4">
            <Button variant="outline" onClick={onClose}>Cancel</Button>
            <Button onClick={handleSubmit} disabled={state.loading}>
              {state.loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Uploading...
                </>
              ) : 'Upload'}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

interface ClusterFormState {
  title: string;
  desc: string;
  loading: boolean;
}

const initialClusterState: ClusterFormState = {
  title: '',
  desc: '',
  loading: false,
};

export const ClusterDialog: React.FC<DialogProps> = ({ onClose }) => {
  const [state, setState] = useState<ClusterFormState>(initialClusterState);

  const handleSubmit = async () => {
    if (!state.title || !state.desc) {
      toast.error('Please fill in all required fields');
      return;
    }

    setState(prev => ({ ...prev, loading: true }));
    try {
      const { error } = await createCluster(state.title, state.desc);
      if (error) throw error;
      toast.success('Cluster created successfully');
      setState(initialClusterState);
      onClose();
    } catch (error) {
      console.error('Failed to create cluster:', error);
      toast.error('Failed to create cluster');
    } finally {
      setState(prev => ({ ...prev, loading: false }));
    }
  };

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Create New Cluster</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="title">Title *</Label>
            <Input
              id="title"
              value={state.title}
              onChange={e => setState(prev => ({ ...prev, title: e.target.value }))}
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="desc">Description *</Label>
            <Textarea
              id="desc"
              className="resize-none"
              value={state.desc}
              onChange={e => setState(prev => ({ ...prev, desc: e.target.value }))}
              required
            />
          </div>

          <div className="flex justify-end space-x-2 pt-4">
            <Button variant="outline" onClick={onClose}>Cancel</Button>
            <Button onClick={handleSubmit} disabled={state.loading}>
              {state.loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Creating...
                </>
              ) : 'Create Cluster'}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export const VisualSearchDialog: React.FC<DialogProps> = ({ onClose }) => {

  const [loading, setLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const urlInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      await uploadImage(file);
    }
  };

  const uploadImage = async (file: File) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    try {
      const {data, error} = await createElement(formData);
      console.log(data, error); //to be used properly
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await uploadImage(file);
    }
  };

  const handleSearch = async () => {
    if (!imageUrl) return;

    setLoading(true);
    try {
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      const file = new File([blob], 'image.jpg', { type: blob.type });
      await uploadImage(file);
    } catch (error) {
      console.error('Download failed:', error);
      setLoading(false);
    }
  };

  return (
    <Dialog open={true} onOpenChange={onClose}>

      <DialogContent className="sm:max-w-md p-6">
        <DialogHeader>
          <DialogTitle>Visual Search</DialogTitle>

        </DialogHeader>
        <div
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:border-blue-500 transition-colors"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="w-16 h-16 flex items-center justify-center">
              <Image src="/assets/upload.png" alt="upload" width={50} height={50} />
            </div>
            <p className="text-lg font-semibold text-gray-600">
              Drag & drop an image
            </p>
            <Button
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="w-full mt-4"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                'Choose File'
              )}
            </Button>
          </div>
        </div>

        <Separator className="my-4" />

        <div className="flex gap-2">
          {/* <Label htmlFor="image-url">Image URL</Label> */}
          <Input
            ref={urlInputRef}
            id="image-url"
            type="url"
            value={imageUrl}
            onChange={(e) => setImageUrl(e.target.value)}
            placeholder="https://example.com/image.jpg"
          />
          <Button
            onClick={handleSearch}
            disabled={!imageUrl || loading}
            className=""
          >
            Search
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

import Image from 'next/image';
import Link from 'next/link';
import Masonry from '@/components/masonry-layout';

interface Image {
  id: number;
  url: string;
  title: string
  desc: string;
  alt: string;
  liked: boolean;
}

type VisualSearchProps = {
  image: Image;
  similar: Image[];
};

export default function VisualSearch({ image, similar }: VisualSearchProps) {
  return (
    <div className="container mx-auto px-4 py-6">
      <div className="flex flex-col lg:flex-row gap-8">
        {/* Featured Image Section */}
        <div className="lg:w-2/5 shrink-0">
          <div className="sticky top-20">
            <div className="rounded-xl overflow-hidden shadow-xl">
              <div className="aspect-w-4 aspect-h-3 relative">
                <Image
                  src={image.url}
                  alt={image.title}
                  className="object-cover transition-transform hover:scale-105 duration-300"
                  width={500} // Adjust width as needed
                  height={500} // Adjust height as needed
                />
              </div>
            </div>
            <div className="mt-6 space-y-4">
              <h1 className="text-2xl font-bold">{image.title}</h1>
              <p className="text-muted-foreground">{image.desc}</p>
              <Link
                href="/home"
                className="inline-block text-primary hover:text-primary/80 underline-offset-4 hover:underline"
              >
                View source
              </Link>
            </div>
          </div>
        </div>

        {/* Masonry Grid Section */}
        <div className="lg:w-3/5">
          <h1 className="text-2xl font-bold">Related Images</h1>
          <Masonry initialImages={similar} />
        </div>
      </div>
    </div>
  );
}
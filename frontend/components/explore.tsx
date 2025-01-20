'use client'

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { Card, CardContent } from '@/components/ui/card';
import { useInView } from 'react-intersection-observer';


type CollectionInfo = {
  title: string;
  user: string;
  thumbnail: string;
};

type ExploreProps = {
  items: CollectionInfo[];
};

const Explore: React.FC<ExploreProps> = ({ items }) => {
  const [exploreItems, setExploreItems] = useState<CollectionInfo[]>(items);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [ref, inView] = useInView({
    threshold: 0,
  });

  useEffect(() => {
    if (inView && !loading) {
      loadMoreItems();
    }
  }, [inView]);

  const loadMoreItems = async () => {
    setLoading(true);
    const { data } = [] as any;
    setExploreItems([...exploreItems, ...data]);
    setPage(page + 1);
    setLoading(false);
  };

  return (
    <div className="container mx-auto px-4 py-2">
      <h1 className="text-3xl font-bold text-center my-4">Discover</h1>
      {/* Brief Description */}
      <p className="text-center text-sm text-gray-500 dark:text-gray-400 mb-10">
        Explore a curated selection of captivating images.
      </p>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 w-full">
        {exploreItems.map((item, index) => (
          <Card 
            key={index}
            className="rounded-none relative aspect-square group overflow-hidden"
          >
            <CardContent className="p-0 h-full">
              <Link href={`/c/${item.title}`} className="block h-full">
                <img 
                  src={item.thumbnail} 
                  alt={item.title} 
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                />
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-between p-4">
                  <h3 className="text-white text-xl font-bold text-center">
                    {item.title}
                  </h3>
                  <p className="text-white text-sm text-center">
                    by {item.user}
                  </p>
                </div>
              </Link>
            </CardContent>
          </Card>
        ))}
      </div>
      {loading && <p className="text-center mt-4">Loading more...</p>}
      <div ref={ref} className="h-10" />
      {!loading && <p className="text-center mt-4">You have reached the end!</p>}
    </div>
  );
};

export default Explore;


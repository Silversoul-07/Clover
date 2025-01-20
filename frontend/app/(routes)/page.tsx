import React from 'react';
import { Metadata } from 'next';
import { fetchFeed } from '@/lib/api';
import Masonry from "@/components/masonry-layout";

export const metadata: Metadata = {
  title: 'Home',
  description: 'A gallery of images',
};

export default async function HomePage() {
  const { data, error } = await fetchFeed();
  if (error) throw error;
  const { elements } = data; // { elements, Tags }

  return <Masonry initialImages={elements} />;

}
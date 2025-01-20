import React from 'react';
import type { Metadata } from 'next'
import { fetchSimilar } from '@/lib/api';
import VisualSearch from '@/components/visual-search';

type Props = {
  params: Promise<{ id: string }>
}

export const metadata: Metadata = {
  title: 'Visual Search',
  description: 'A gallery of images',
};

export default async function VisualSearchPage({ params }: Props) {
  const { id } = (await params);
  const { data, error } = await fetchSimilar(undefined, id);
  if (error) throw new Error(error);
  const { elements, ...image } = data;
  return (
    <VisualSearch image={image} similar={elements} />
  );
}

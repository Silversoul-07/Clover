import React from 'react';
import type { Metadata } from 'next'
import ImageView from '@/components/image-view';
import { fetchElement } from '@/lib/api';

type Props = {
  params: Promise<{ id: string }>
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>
}

export const metadata: Metadata = {
  title: 'Element',
  description: 'Collection page',
};

export default async function ImageViewPage({ params }: Props) {
  const { id } = (await params);
  const { data, error } = await fetchElement(id);
  if (error) throw new Error(error);
  const { similar, ...image } = data;

  return (
    <ImageView image={image.url} title={"Dummy Caption"} similar={similar} />
  )
}

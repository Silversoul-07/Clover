import React from 'react';
import type { Metadata } from 'next'
import Masonry from '@/components/masonry-layout';
import { fetchSearch } from '@/lib/api';

type Props = {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>
}

export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const title = (await searchParams).q  as string;

  return {
    title: title,
    description: "Collection page",
  }
}

export default async function HomePage({ searchParams }: Props) {
  const query = (await searchParams).q as string;
  const { data, error } = await fetchSearch(query);
  if (error) throw error;
  const { elements } = data;


  return <Masonry initialImages={elements} />;

}
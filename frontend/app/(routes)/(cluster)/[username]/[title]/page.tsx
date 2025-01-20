import ImageCluster from "@/components/cluster";
import Masonry from "@/components/masonry-layout";
import { fetchCluster } from "@/lib/api";
import type { Metadata } from 'next'

type Props = {
  params: Promise<{ username:string, title: string }>
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const title = (await params).title

  return {
    title: title,
    description: "Collection page",
  }
}

export default async function CollectionPage({ params }: Props) {
  const { username, title } = await params
  console.log(username, title)
  const {data, error} = await fetchCluster(username, title);
  if (error) {
    throw new Error(error);
  }
  const {elements, ...info} = data;
  return (
    <div>
      <ImageCluster clusterInfo={info} />
      <Masonry initialImages={elements} />
    </div>
  )
}
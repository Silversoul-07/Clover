import ExploreComponent, { Category } from "@/components/explore"

const categories: Category[] = [
  { title: "Nature", image: "https://picsum.photos/seed/nature/400/300" },
  { title: "Architecture", image: "https://picsum.photos/seed/arch/400/300" },
  { title: "Travel", image: "https://picsum.photos/seed/travel/400/300" },
  { title: "Urban", image: "https://picsum.photos/seed/urban/400/300" },
  { title: "Food", image: "https://picsum.photos/seed/food/400/300" },
  { title: "People", image: "https://picsum.photos/seed/people/400/300" },
  { title: "Technology", image: "https://picsum.photos/seed/tech/400/300" },
  { title: "Animals", image: "https://picsum.photos/seed/animals/400/300" },
]

const popularImages = Array(24)
  .fill(null)
  .map((_, i) => `https://picsum.photos/seed/${i}/300/200`)

const searchTags = [
  "Landscape",
  "Portrait",
  "Street",
  "Wildlife",
  "Urban",
  "Minimal",
  "Abstract",
  "Black & White",
  "Macro",
  "Night",
  "Aerial",
  "Underwater",
]

export default function ExplorePage() {
  return <ExploreComponent categories={categories} popularImages={popularImages} searchTags={searchTags} />
}


"use client"

import type React from "react"
import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

export interface Category {
  title: string
  image: string
}

export interface ExploreComponentProps {
  categories: Category[]
  popularImages: string[]
  searchTags: string[]
}

const ExploreComponent: React.FC<ExploreComponentProps> = ({ categories, popularImages, searchTags }) => {
  const [visibleImages, setVisibleImages] = useState<number>(12)
  const [visibleCategories, setVisibleCategories] = useState<number>(4)

  const loadMoreImages = () => {
    setVisibleImages((prev) => Math.min(prev + 8, popularImages.length))
  }

  const loadMoreCategories = () => {
    setVisibleCategories((prev) => Math.min(prev + 4, categories.length))
  }

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-12 my-6">
      {/* Categories */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="space-y-6"
      >
        <div className="text-center space-y-2">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 bg-clip-text text-transparent">
            Explore Categories
          </h2>
          <p className="text-gray-600 dark:text-gray-400">Discover our curated collection of photography themes</p>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <AnimatePresence>
            {categories.slice(0, visibleCategories).map((category, index) => (
              <motion.div
                key={category.title}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <Card className="relative h-48 overflow-hidden group cursor-pointer">
                  <img
                    src={category.image || "/placeholder.svg"}
                    alt={category.title}
                    className="absolute inset-0 w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent">
                    <h3 className="text-xl font-bold absolute bottom-4 left-4 text-white">{category.title}</h3>
                  </div>
                </Card>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
        {visibleCategories < categories.length && (
          <div className="text-center mt-8">
            <Button
              onClick={loadMoreCategories}
              className="bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 text-white hover:opacity-90 transition-all duration-300"
            >
              View More Categories
            </Button>
          </div>
        )}
      </motion.section>

      {/* Popular Images Grid */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="space-y-6"
      >
        <div className="text-center space-y-2">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 bg-clip-text text-transparent">
            Popular Now
          </h2>
          <p className="text-gray-600 dark:text-gray-400">Trending images loved by our community</p>
        </div>
        <div className="columns-2 sm:columns-3 lg:columns-4 gap-4">
          <AnimatePresence>
            {popularImages.slice(0, visibleImages).map((image, index) => (
              <motion.div
                key={image}
                className="break-inside-avoid mb-4"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <div className="relative overflow-hidden rounded-lg group">
                  <img
                    src={image || "/placeholder.svg"}
                    alt={`Popular ${index + 1}`}
                    className="w-full object-cover transition-transform duration-300 group-hover:scale-110"
                  />
                  <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-opacity duration-300" />
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
        {visibleImages < popularImages.length && (
          <div className="text-center mt-8">
            <Button
              onClick={loadMoreImages}
              className="bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 text-white hover:opacity-90 transition-all duration-300"
            >
              Load More Images
            </Button>
          </div>
        )}
      </motion.section>

      {/* Search Tags */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="flex flex-col items-center space-y-4"
      >
        <h2 className="text-3xl font-bold bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 bg-clip-text text-transparent">
          Popular Tags
        </h2>
        <div className="flex flex-wrap justify-center gap-2 max-w-3xl">
          {searchTags.map((tag, index) => (
            <motion.span
              key={tag}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className="px-4 py-2 bg-gradient-to-r from-pink-500/30 via-purple-500/30 to-blue-500/30 
                hover:from-pink-500/40 hover:via-purple-500/40 hover:to-blue-500/40 
                rounded-full text-sm cursor-pointer transition-colors"
            >
              {tag}
            </motion.span>
          ))}
        </div>
      </motion.section>
    </div>
  )
}

export default ExploreComponent
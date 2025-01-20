from typing import List, Optional
import os
import numpy as np
from pydantic import BaseModel
from typing import Union

from pymilvus import (
    connections, 
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType
)



class SearchResult(BaseModel):
    """Represents a search result with image ID and similarity score."""
    image_id: str
    score: float

class MilvusService:
    """Service for managing vector embeddings in Milvus."""
    
    def __init__(self, collection_name: str = "images") -> None:
        """Initialize Milvus service and collection."""
        self.collection_name = collection_name
        
    def create(self) -> None:
        """Initialize Milvus connection and create collection if not exists."""
        try:
            connections.connect(
                alias="default",
                host=os.getenv("MILVUS_HOST", "milvus"),
                port=os.getenv("MILVUS_PORT", "19530")
            )
            
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                return
                
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
                FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            ]
            
            schema = CollectionSchema(fields)
            collection = Collection(name=self.collection_name, schema=schema)
            
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",  # Fixed typo
                "params": {"nlist": 128}
            }
            
            for field in ["image_embedding", "text_embedding"]:  # Removed invalid field
                collection.create_index(field, index_params)
            
            self.collection = collection
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Milvus: {str(e)}")

    def insert(self, 
               image_id: str, 
               image_embedding: List[float], 
               text_embedding: Optional[List[float]] = None) -> None:
        """Insert image and text embeddings into collection."""
        try:
            self.collection.insert([
                [image_id],
                image_embedding,
                text_embedding,
            ])
            self.collection.flush()
        except Exception as e:
            raise RuntimeError(f"Failed to insert embeddings: {str(e)}")
        
    def get(self, id: str) -> Optional[dict]:
        """Retrieve image and text embeddings by ID."""
        try:
            self.collection.load()
            
            results = self.collection.query(expr=f"id == '{id}'", output_fields=["id", "image_embedding", "text_embedding"])
            if not results:
                return None
            
            return {
                "id": results[0]["id"],
                "image_embedding": results[0]["image_embedding"],
                "text_embedding": results[0]["text_embedding"],
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get embeddings: {str(e)}")

    
    def search(self, 
               query_embedding: Union[np.ndarray],
               field: str,
               limit: int = 50) -> List[SearchResult]:
        """Search for similar embeddings in the specified field."""
        try:
            if field not in ["image_embedding", "text_embedding"]:
                raise ValueError("Invalid field. Must be 'image_embedding' or 'text_embedding'")
            # Convert torch tensor to numpy array if necessary
            # if isinstance(query_embedding, torch.Tensor):
                # query_embedding = query_embedding.detach().cpu().numpy()
            if isinstance(query_embedding[0], np.float32):
                query_embedding = np.array([query_embedding], dtype=np.float32)

            # Ensure query_embedding is a numpy array of floats
            print(type(query_embedding), query_embedding.dtype, query_embedding.shape)
            if not isinstance(query_embedding, np.ndarray) or query_embedding.dtype != np.float32:
                raise ValueError("query_embedding must be a numpy array of float32")
    
            self.collection.load()
    
            results = self.collection.search(
                data=query_embedding,
                anns_field=field,
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=limit,
                output_fields=["id"]
            )
    
            return [
                SearchResult(
                    image_id=hit.id,
                    score=float(hit.score)
                )
                for hits in results
                for hit in hits
            ]
        except Exception as e:
            print(e)
            raise RuntimeError(f"Failed to search embeddings: {str(e)}")
        
    async def search_similar_images(
        self,
        embedding: np.ndarray,
        limit: int = 20,
        distance_threshold: float = 0.8
    ) -> List[int]:
        search_params = {
            "metric_type": "IP",  # Inner Product similarity
            "params": {"nprobe": 10},
        }
        
        results = self.collection.search(
            data=[embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=None
        )

        # Filter by distance threshold and extract IDs
        similar_ids = []
        for hits in results:
            for hit in hits:
                if hit.distance >= distance_threshold:
                    similar_ids.append(hit.id)

        return similar_ids
        
    def __del__(self):
        """Cleanup Milvus connection on object destruction."""
        try:
            connections.disconnect("default")
        except:
            pass

    async def search_similar_images(
        self,
        embedding: np.ndarray,
        limit: int = 20,
        distance_threshold: float = 0.8
    ) -> List[int]:
        search_params = {
            "metric_type": "IP",  # Inner Product similarity
            "params": {"nprobe": 10},
        }
        
        results = self.collection.search(
            data=[embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=None
        )

        # Filter by distance threshold and extract IDs
        similar_ids = []
        for hits in results:
            for hit in hits:
                if hit.distance >= distance_threshold:
                    similar_ids.append(hit.id)

        return similar_ids
    



milvus_service = MilvusService()
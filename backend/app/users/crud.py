from sqlalchemy.orm import Session
from sqlalchemy.sql import func 
from .models import Users, Clusters, Elements, CacheEntry, UserInteraction, InteractionType
from . import schemas
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import json
import numpy as np
from typing import List
from ..milvus import MilvusService


async def is_username_exists(db: Session, username: str):
    user = db.query(Users).where(Users.username == username).first()
    if user:
        return True
    return False

async def get_user_by_id(db: Session, user_id: str):
    user = db.query(Users).where(Users.id == user_id).first()
    if not user:
        return None
    return user

async def get_user_by_username(db: Session, username: str, user_id: Optional[str]=None):
    user = db.query(Users).where(Users.username == username).first()
    if not user:
        return None
    return user

async def get_cluster_by_title(db: Session, title: str, user_id: str):
    cluster = db.query(Clusters).where(Clusters.title == title, Clusters.user_id == user_id).first()
    if not cluster:
        return None
    return cluster


async def get_user_clusters(db: Session, user_id: str, username: str):
    clusters = db.query(Clusters).filter(Clusters.user_id == user_id).all()
    
    for cluster in clusters:
        if not cluster.thumbnail or 'placeholder' in cluster.thumbnail:
            random_element = db.query(Elements).filter(Elements.cluster_id == cluster.id).order_by(func.random()).first()
            if random_element:
                cluster.thumbnail = random_element.url
            else:
                cluster.thumbnail = "https://via.placeholder.com/150"
    
    db.commit()
    
    return [schemas.ClusterBase(**cluster.__dict__, count=db.query(Elements).filter(Elements.cluster_id == cluster.id).count()) for cluster in clusters if cluster.title]

async def get_user_stats(db: Session, user_id: str):
    # clusters = db.query(Clusters).where(Clusters.user_id == user_id).count()
    elements = db.query(Elements).where(Elements.user_id == user_id).count()
    likes = db.query(Users).where(Users.id == user_id).first().likes
    return schemas.UserStats(clusters=0, elements=elements, likes=likes)

async def get_elements_in_cluster(db: Session, cluster_id: str, user_id: str):
    # cluster = db.select(Clusters).where(Clusters.id == cluster_id, Clusters.user_id == user_id).first()
    # if not cluster:
    #     return None
    elements = db.query(Elements).where(Elements.cluster_id == cluster_id).all()
    return [schemas.ElementBase(**element.__dict__) for element in elements]

async def get_element(db: Session, element_id: str):
    element = db.query(Elements).where(Elements.id == element_id).first()
    if not element:
        return None
    return schemas.ElementBase(**element.__dict__)  

async def get_element_by_hash(db: Session, hash: str):
    element = db.query(Elements).where(Elements.hash == hash).first()
    if not element:
        return None
    return schemas.ElementBase(**element.__dict__)
    
async def insert(db:Session, entity: Any):
    db.add(entity)
    db.commit()
    db.refresh(entity)

async def init_clusters(db: Session, kwargs: Dict[str, Any]):
    user = db.query(Users).where(Users.username == kwargs["username"]).first()
    clusters = kwargs["clusters"]
    for cluster in clusters:
        cluster["user_id"] = user.id
        db.add(Clusters(**cluster))
    db.commit()
    db.refresh(user)
    return user.id

class CacheManager:
    def __init__(self, db_session):
        self.db = db_session

    async def get(self, key: str) -> Optional[Any]:
        cache_entry = self.db.query(CacheEntry).filter(CacheEntry.key == key).first()
        if cache_entry and not cache_entry.is_expired:
            return json.loads(cache_entry.value)
        return None

    async def set(self, key: str, value: Any, expire_in: Optional[int] = None):
        expires_at = None
        if expire_in:
            expires_at = datetime.utcnow() + timedelta(seconds=expire_in)

        cache_entry = self.db.query(CacheEntry).filter(CacheEntry.key == key).first()
        if cache_entry:
            cache_entry.value = json.dumps(value)
            cache_entry.expires_at = expires_at
        else:
            cache_entry = CacheEntry(
                key=key,
                value=json.dumps(value),
                expires_at=expires_at
            )
            self.db.add(cache_entry)
        
        self.db.commit()

    async def delete(self, key: str):
        self.db.query(CacheEntry).filter(CacheEntry.key == key).delete()
        self.db.commit()
        

class RecommendationEngine:
    def __init__(self, db: Session, milvus_client: MilvusService):
        self.db = db
        self.milvus_client = milvus_client
        self.interaction_weights = {
            InteractionType.like: 3.0,
            InteractionType.save: 2.0,
            InteractionType.click: 1.0
        }

    def _get_user_interactions(self, user_id: str, days: int = 30) -> Dict[int, float]:
        """Get user's recent interactions with weights"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        interactions = self.db.query(
            UserInteraction.element_id,
            UserInteraction.interaction_type,
            UserInteraction.timestamp
        ).filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= cutoff_date
        ).all()

        # Calculate weighted scores
        element_scores = {}
        for element_id, interaction_type, timestamp in interactions:
            # Add recency factor (1.0 to 0.5 based on age)
            days_old = (datetime.utcnow() - timestamp).days
            recency_factor = 1.0 - (days_old / (days * 2))
            recency_factor = max(0.5, recency_factor)
            
            score = self.interaction_weights[interaction_type] * recency_factor
            element_scores[element_id] = element_scores.get(element_id, 0) + score
            
        return element_scores

    async def get_recommendations(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[Elements]:
        """Get personalized recommendations for user"""
        
        # Get user interaction history
        interaction_scores = self._get_user_interactions(user_id)
        
        if not interaction_scores:
            # Fall back to popular items if no interaction history
            return self._get_popular_elements(limit, offset)

        # Get embeddings for interacted elements
        interacted_elements = self.db.query(Elements).filter(
            Elements.id.in_(interaction_scores.keys())
        ).all()

        # Aggregate embeddings weighted by interaction scores
        embeddings = []
        for element in interacted_elements:
            if element.analysis and 'embedding' in element.analysis:
                score = interaction_scores[element.id]
                embeddings.append(
                    np.array(element.analysis['embedding']) * score
                )

        if not embeddings:
            return self._get_popular_elements(limit, offset)

        # Average the weighted embeddings
        query_embedding = np.mean(embeddings, axis=0)

        # Search similar vectors in Milvus
        similar_ids = await self.milvus_client.search_similar_images(
            query_embedding,
            limit=limit * 2  # Get extra to filter out already seen
        )

        # Filter out elements user has already interacted with
        similar_ids = [id for id in similar_ids if id not in interaction_scores]

        # Get recommended elements
        recommendations = self.db.query(Elements).filter(
            Elements.id.in_(similar_ids[:limit])
        ).offset(offset).all()

        return recommendations

    def _get_popular_elements(self, limit: int, offset: int) -> List[Elements]:
        """Fallback to popular items"""
        return self.db.query(Elements).join(
            UserInteraction
        ).group_by(
            Elements.id
        ).order_by(
            func.count(UserInteraction.id).desc()
        ).offset(offset).limit(limit).all()
import enum
from sqlalchemy import Column, Float, Integer, String, Boolean, DateTime, ForeignKey, Enum, Text, UniqueConstraint, JSON
from sqlalchemy.sql import func
from sqlalchemy.schema import Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from ..config import get_env
from ..database import Base
import uuid

env = get_env()
import uuid
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.sql import func
from ..database import Base

# User model
class UserProfile(Base):
    """Stores public user profile details."""
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=False), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    bio = Column(String(500))
    avatar = Column(String(255))
    is_private = Column(Boolean, default=False)
    likes_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class UserAuth(Base):
    """Stores sensitive authentication information."""
    __tablename__ = 'user_auth'

    id = Column(UUID(as_uuid=False), ForeignKey('user_profiles.id', ondelete='CASCADE'), primary_key=True, nullable=False)
    username = Column(String(50), nullable=False, unique=True)
    password = Column(String(255), nullable=False)

# Cluster model
class Clusters(Base):
    """Content organization units that group related elements together."""
    __tablename__ = 'clusters'

    id = Column(UUID(as_uuid=False), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    href = Column(String(255), nullable=False)
    thumbnail_url = Column(String(255))
    is_private = Column(Boolean, default=False)
    user_id = Column(UUID(as_uuid=False), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index('idx_clusters_user_created', 'user_id', 'created_at'),
        Index('idx_clusters_title_trgm', 'title', postgresql_using='gin', postgresql_ops={'title': 'gin_trgm_ops'}),
    )

class ClusterElements(Base):
    """Association table linking clusters to elements for saved/liked images."""
    __tablename__ = 'cluster_elements'

    cluster_id = Column(UUID(as_uuid=False), ForeignKey('clusters.id', ondelete='CASCADE'), primary_key=True, nullable=False)
    element_id = Column(UUID(as_uuid=False), ForeignKey('elements.id', ondelete='CASCADE'), primary_key=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

# Element model 
class Elements(Base):
    """Individual content items that can be organized into clusters."""
    __tablename__ = 'elements'

    id = Column(UUID(as_uuid=False), primary_key=True, default=uuid.uuid4)
    url = Column(String(2048), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    content_hash = Column(String(64), unique=True, nullable=False)
    placeholder = Column(String(255))
    source_id = Column(UUID(as_uuid=False), ForeignKey('sources.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(UUID(as_uuid=False), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index('idx_elements_user_cluster', 'user_id'),
        Index('idx_elements_content_hash', 'content_hash'),
        Index('idx_elements_title_trgm', 'title', postgresql_using='gin', postgresql_ops={'title': 'gin_trgm_ops'}),
    )

# Tag model
class Tags(Base):
    """Category labels that can be applied to elements."""
    __tablename__ = "tags"
    
    id = Column(UUID(as_uuid=False), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class ElementTags(Base):
    """Association table linking elements to tags with additional metadata."""
    __tablename__ = "element_tags"
    
    element_id = Column(UUID(as_uuid=False), ForeignKey('elements.id', ondelete='CASCADE'), primary_key=True, nullable=False)
    tag_id = Column(UUID(as_uuid=False), ForeignKey('tags.id', ondelete='CASCADE'), primary_key=True, nullable=False)
    score = Column(Float, nullable=False, default=0.0)
    thumbnail = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

# User interactions Model
class InteractionType(enum.Enum):
    """Valid types of user interactions with elements."""
    like = 'like'
    save = 'save'
    click = 'click'

class UserInteractions(Base):
    """Records of user interactions with elements."""
    __tablename__ = 'user_interactions'

    id = Column(UUID(as_uuid=False), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=False), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    element_id = Column(UUID(as_uuid=False), ForeignKey('elements.id', ondelete='CASCADE'), nullable=False)
    interaction_type = Column(Enum(InteractionType), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    time_spent_seconds = Column(Integer, default=0)

    __table_args__ = (
        Index('idx_interactions_user_element', 'user_id', 'element_id'),
        Index('idx_interactions_type_timestamp', 'interaction_type', 'timestamp'),
    )

# Embeddings Model
class ImageEmbeddings(Base):
    """Stores image embeddings for elements."""
    __tablename__ = "image_embeddings"
    
    id = Column(UUID(as_uuid=False), primary_key=True, default=uuid.uuid4)
    image_embedding = Column(Vector(env.imgEmdSize))
    element_id = Column(UUID(as_uuid=False), ForeignKey('elements.id', ondelete='CASCADE'), nullable=False)
    version = Column(Integer, nullable=False, default=env.embedding_version)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class TextEmbeddings(Base):
    """Stores text embeddings for elements."""
    __tablename__ = "text_embeddings"
    
    id = Column(UUID(as_uuid=False), primary_key=True, default=uuid.uuid4)
    text_embedding = Column(Vector(env.txtEmdSize))
    element_id = Column(UUID(as_uuid=False), ForeignKey('elements.id', ondelete='CASCADE'), nullable=False)
    version = Column(Integer, nullable=False, default=env.embedding_version)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

# Miscellanous Models
class Sources(Base):
    """External content sources with domain tracking."""
    __tablename__ = 'sources'

    id = Column(UUID(as_uuid=False), primary_key=True, default=uuid.uuid4)
    url = Column(Text, unique=True, nullable=False)
    domain = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index('idx_sources_domain_lower', func.lower('domain')),
        Index('idx_sources_url_hash', 'url', postgresql_using='hash'),
    )
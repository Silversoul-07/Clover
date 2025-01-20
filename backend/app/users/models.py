import enum
from numpy import save
from sqlalchemy import Column, String, Boolean, JSON, DateTime, ForeignKey, BigInteger, Text, Enum, Integer
from sqlalchemy.dialects.postgresql import UUID
from ..database import Base
import uuid
from datetime import datetime
from sqlalchemy.sql import func

def rand_id() -> int:
    return uuid.uuid4().time

class Users(Base):
    __tablename__ = 'users'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    name = Column(String)
    username = Column(String, unique=True)
    password = Column(String)
    bio = Column(String)
    avatar = Column(String)
    notifications = Column(JSON)
    private = Column(Boolean, default=False)
    likes = Column(BigInteger, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class Clusters(Base):
    __tablename__ = 'clusters'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    title = Column(String)
    desc = Column(String)
    href = Column(String)
    thumbnail = Column(String, nullable=True)  # Added thumbnail attribute
    private = Column(Boolean, default=False)
    user_id = Column(UUID, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)

class Elements(Base):
    __tablename__ = 'elements'

    id = Column(BigInteger, primary_key=True, default=rand_id)
    url = Column(String)
    title = Column(String)
    desc = Column(String)
    hash = Column(String)
    placeholder = Column(String)
    analysis = Column(JSON)
    source = Column(String)
    cluster_id = Column(UUID, ForeignKey('clusters.id'))
    user_id = Column(UUID, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)

class ClusterFollowers(Base):
    __tablename__ = 'followers'

    id = Column(BigInteger, primary_key=True, default=rand_id)
    user_id = Column(UUID, ForeignKey('users.id'))
    cluster_id = Column(UUID, ForeignKey('clusters.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    

class CacheEntry(Base):
    __tablename__ = "cache_entries"

    key = Column(String, primary_key=True, index=True)
    value = Column(Text)
    expires_at = Column(DateTime, nullable=True)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
class InteractionType(enum.Enum):
    like = 'like'
    save = 'save'
    click = 'click'

class UserInteraction(Base):
    __tablename__ = 'user_interactions'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID, ForeignKey('users.id'), nullable=False)
    element_id = Column(BigInteger, ForeignKey('elements.id'), nullable=False)
    interaction_type = Column(Enum(InteractionType), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    time_spent = Column(Integer)  # Time in seconds

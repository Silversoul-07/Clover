from functools import wraps
from typing import Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Any
from sqlalchemy import Column, String, Text, DateTime
from .database import Base
from sqlalchemy.orm import Session
import json
from fastapi.encoders import jsonable_encoder

from hashlib import sha256
import json
from typing import Any

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


class CacheManager:
    def __init__(self, session: Session):
        self.db = session

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


def generate_cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate deterministic cache key."""
    def serialize_arg(arg: Any) -> str:
        if isinstance(arg, dict):
            return json.dumps(arg, sort_keys=True)
        return str(arg)

    components = [
        kwargs.get('func_name', ''),
        ':'.join(serialize_arg(arg) for arg in args if arg is not None),
        ':'.join(f"{k}={serialize_arg(v)}" for k, v in sorted(kwargs.items()) 
                if k != 'func_name' and v is not None and not hasattr(v, '__dict__'))
    ]
    
    key = ':'.join(filter(None, components))
    return f"{sha256(key.encode()).hexdigest()}"

def cached(expire_in: Optional[int] = None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            db = kwargs.get('db')
            cache = CacheManager(db)
            
            # Generate cache key
            cache_key = generate_cache_key(
                func_name=func.__name__,
                *args,
                **kwargs
            )
            
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                print('cache hit')
                return cached_value
                
            print('cache miss')
            result = await func(*args, **kwargs)
            result = jsonable_encoder(result)
            await cache.set(cache_key, result, expire_in)
            return result
            
        return wrapper
    return decorator
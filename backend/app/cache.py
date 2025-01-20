from datetime import datetime, timedelta
from typing import Optional, Any
import json
from sqlalchemy import Column, String, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from functools import wraps

# Database setup
SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/dbname"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Cache manager




@app.get("/users/{user_id}", response_model=UserResponse)
@cached(expire_in=300)  # Cache for 5 minutes
async def get_user(user_id: int):
    # Simulate database query
    user = {
        "id": user_id,
        "name": "John Doe",
        "email": "john@example.com"
    }
    return UserResponse(**user)

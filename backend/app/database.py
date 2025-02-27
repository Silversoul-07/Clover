from sqlite3 import connect
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import text
from .config import get_env  

env = get_env()

engine = create_engine(env.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db(Base, engine):
    with engine.connect() as connection:
        connection.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        connection.execute(text('CREATE EXTENSION IF NOT EXISTS pg_trgm'))
        # connection.execute(text("DROP TABLE tags CASCADE"))
        connection.commit()
    # Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Database initialized")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
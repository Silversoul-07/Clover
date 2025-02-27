from . import models, schemas
from sqlalchemy.orm import Session

# Users CRUD
async def check_user_exists(db: Session, username: str):
    user = db.query(models.UserAuth).where(models.UserAuth.username == username).first()
    if not user:
        return None
    return user

# Cluster CRUD
# Element CRUD
# Tag CRUD
# Embedding CRUD
# Miscellanouse CRUD
from sqlalchemy.orm import Session
from sqlalchemy.sql import func 
from . import schemas, models
from typing import Coroutine, Optional, Any, Dict, Tuple, List, Callable
from uuid import UUID
from sqlalchemy import select
import tldextract
from sqlalchemy.exc import IntegrityError

async def create_user_interaction(db: Session, user_id: str, element_id: str, interaction_type: InteractionType, time_spent_seconds: int = 0):
    interaction = UserInteractions(
        user_id=user_id,
        element_id=element_id,
        interaction_type=interaction_type,
        time_spent_seconds=time_spent_seconds
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    return interaction

async def add_like(db: Session, user_id: str, element_id: str):
    # add image to cluster named "liked"
    liked_cluster = await get_cluster_by_title(db, "liked", user_id)
    if not liked_cluster:
        return None
    element = await get_element(db, element_id)
    if not element:
        return None
    element.cluster_id = liked_cluster.id
    await update(db, element)
    

async def init_clusters(db: Session, user_id: str, username: str):
    """
    Initialize default clusters ("profile" and "liked") for a user.
    If the clusters already exist, they are left unchanged.
    """
    profile_cluster = Clusters(
            title="profile",
            description="Your profile",
            href=f"/{username}/profile",
            is_private=True,
            user_id=user_id,
        )
    await insert(db, profile_cluster)
    
    liked_cluster = Clusters(
            title="liked",
            description="Liked elements",
            href=f"/{username}/liked",
            is_private=True,
            user_id=user_id,
        )
    await insert(db, liked_cluster)

async def get_all_element_ids(db: Session):
    elements = db.query(Elements).all()
    return [element.id for element in elements]

async def get_cluster_by_title(db: Session, title: str, user_id: str):
    cluster = db.query(Clusters).where(Clusters.title == title, Clusters.user_id == user_id).first()
    if not cluster:
        return None
    return cluster

async def insert_embedding(db: Session, element_id: int, text_embedding: List[float], image_embedding: List[float], version:int) -> None:
    embedding = Embeddings(
        element_id=element_id,
        text_embedding=text_embedding,
        image_embedding=image_embedding,
        version=version
    )
    db.add(embedding)
    db.commit()
    db.refresh(embedding)


from sqlalchemy.orm import Session
from typing import List, Tuple
import uuid

async def get_element_tags(db: Session, element_id: uuid.UUID) -> List[Tuple[str, float]]:
    tags = (
        db.query(Tags.name, ElementTags.score)
          .join(ElementTags, ElementTags.tag_id == Tags.id)
          .filter(ElementTags.element_id == element_id)
          .order_by(ElementTags.score.desc())
          .all()
    )
    return tags

async def insert_tag(db: Session, tag_name: str, score: float, element_id: uuid.UUID) -> bool:
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy.exc import IntegrityError

    try:
        # Normalize tag name
        tag_name = tag_name.lower()

        # Check if element-tag association already exists
        existing_association = db.query(ElementTags).join(Tags).filter(
            Tags.name == tag_name,
            ElementTags.element_id == element_id
        ).first()

        if existing_association:
            print(f"Association already exists: {existing_association}")
            return True  # Association already exists

        # Get or create tag
        tag = db.query(Tags).filter(func.lower(Tags.name) == tag_name).first()
        if not tag:
            try:
                # Try to create new tag
                stmt = insert(Tags).values(name=tag_name).on_conflict_do_nothing()
                db.execute(stmt)
                db.flush()
                
                # Get the newly created tag
                tag = db.query(Tags).filter(Tags.name == tag_name).first()
            except IntegrityError:
                print(f"Failed to insert tag: {tag_name}")
                db.rollback()
                # If tag was created concurrently, try to get it again
                tag = db.query(Tags).filter(Tags.name == tag_name).first()

        if not tag:
            print(f"Failed to create or retrieve tag: {tag_name}")
            return False  # Failed to create or retrieve tag

        # Create element-tag association
        element_tag = ElementTags(
            element_id=element_id,
            tag_id=tag.id,
            score=score
        )
        db.add(element_tag)
        db.commit()
        return True

    except Exception as e:
        db.rollback()
        print(f"Failed to insert tag: {str(e)}")
        return False


async def get_tags_for_element(db: Session, element_id: int) -> List[Tuple[str, float]]:
    tags = (
        db.query(Tags.name, ElementTags.score)
          .join(ElementTags, ElementTags.tag_id == Tags.id)
          .filter(ElementTags.element_id == element_id)
          .order_by(ElementTags.score.desc())
          .all()
    )
    return tags

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


async def get_or_create_cluster(db: Session, title: str, user_id: str):
    cluster = db.query(Clusters).where(Clusters.title == title, Clusters.user_id == user_id).first()
    if not cluster:
        cluster = Clusters(title=title, user_id=user_id)
        await insert(db, cluster)
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
    elements = db.query(Elements).where(Elements.user_id == user_id).count()
    likes = db.query(Users).where(Users.id == user_id).first().likes_count
    return schemas.UserStats(clusters=0, elements=elements, likes=likes)

async def get_elements_in_cluster(db: Session, cluster_id: str):
    elements = db.query(Elements).where(Elements.cluster_id == cluster_id).all()
    return [schemas.ElementBase(**element.__dict__) for element in elements]

async def get_element(db: Session, element_id: str):
    element = db.query(Elements).where(Elements.id == element_id).first()
    if not element:
        return None
    return schemas.ElementBase(**element.__dict__)  

async def get_element_by_hash(db: Session, hash: str):
    element = db.query(Elements).where(Elements.content_hash == hash).first()
    if not element:
        return None
    return schemas.ElementBase(**element.__dict__)
    
async def insert(db:Session, entity: Any):
    db.add(entity)
    db.commit()
    db.refresh(entity)

async def update(db: Session, entity: Any):
    db.commit()
    db.refresh(entity)

async def get_or_insert_source(db: Session, url: str):
    source = db.query(Sources).where(Sources.url == url).first()
    if not source:
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"
        source = Sources(url=url, domain=domain)
        await insert(db, source)
    return source


    

async def insert_vector(
        db: Session,
        image_id: str, 
                       image_embedding: List[float], 
                       text_embedding: Optional[List[float]] = None) -> None:
    """Insert embeddings into database"""
    try:
        embedding = Embeddings(
            id=image_id,
            image_embedding=image_embedding,
            text_embedding=text_embedding
        )
        db.add(embedding)
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise RuntimeError(f"Failed to insert embeddings: {str(e)}")

async def vector_search(
        db: Session,
        query_embedding: List[float], 
                       field: str, 
                       limit: int = 50) -> List[schemas.SearchResult]:
    """Search for similar embeddings using pgvector"""
    if field not in ["image_embedding", "text_embedding"]:
        raise ValueError("Invalid field. Must be 'image_embedding' or 'text_embedding'")

    try:
        stmt = select(
            Embeddings.id,
            (Embeddings.__table__.c[field].l2_distance(query_embedding)).label("similarity")
        ).order_by("similarity").limit(limit)
        
        result = await db.execute(stmt)
        return [schemas.SearchResult(image_id=row.id, score=row.similarity) for row in result.fetchall()]
    except Exception as e:
        raise RuntimeError(f"Failed to search embeddings: {str(e)}")
    
async def text_search(
    db: Session,
    query_text: str,
    text2vec: Callable[[str], List[float]],
    limit: int = 10,
    threshold: float = 0.7
) -> List[Tuple[UUID, float]]:
    query_embedding = text2vec(query_text)
    stmt = select(
        Embeddings.element_id,
        Embeddings.text_embedding.cosine_distance(query_embedding).label('distance')
    ).order_by(
        Embeddings.text_embedding.cosine_distance(query_embedding)
    ).limit(limit)
    
    results = db.execute(stmt).fetchall()
    return [(row.element_id, 1 - float(row.distance)) for row in results]
    
async def visual_search(
    db: Session,
    image_bytes: bytes,
    img2vec: Coroutine[bytes, List[float], None],
    limit: int = 10,
    threshold: float = 0.7
) -> List[Tuple[UUID, float]]:
    query_embedding = await img2vec(image_bytes)
    stmt = select(
        Embeddings.element_id,
        Embeddings.image_embedding.cosine_distance(query_embedding).label('distance')
    ).order_by(
        Embeddings.image_embedding.cosine_distance(query_embedding)
    ).limit(limit)
    
    results = db.execute(stmt).fetchall()
    return [(row.element_id, 1 - float(row.distance)) for row in results]

async def get_element_embedding(db: Session, element_id: str):
    """Get embeddings for an element"""
    embedding = db.query(Embeddings).filter(Embeddings.element_id == element_id).first()
    return embedding

async def get_user_likes(db: Session, user_id: str) -> List[str]:
    """Get all elements liked by user"""
    likes = (
        db.query(UserInteractions.element_id)
        .filter(
            UserInteractions.user_id == user_id,
            UserInteractions.interaction_type == InteractionType.like
        )
        .all()
    )
    return [str(like.element_id) for like in likes]

async def get_user_interactions(db: Session, user_id: str) -> List[str]:
    """Get all elements user has interacted with"""
    interactions = (
        db.query(UserInteractions.element_id)
        .filter(UserInteractions.user_id == user_id)
        .distinct()
        .all()
    )
    return [str(interaction.element_id) for interaction in interactions]

async def get_trending_elements(db: Session, limit: int = 20, randomize=False) -> List[Elements]:
    """Get trending elements based on interaction count"""
    if randomize:
        offset = func.floor(func.random() * db.query(UserInteractions).count())
    else:
        offset = 0
    trending = (
        db.query(Elements)
        .join(UserInteractions)
        .group_by(Elements.id)
        .order_by(func.count(UserInteractions.id).desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return trending

async def vector_search(
    db: Session,
    query_embedding: List[float],
    field: str,
    limit: int = 20,
    offset: int = 0,
    exclude_ids: Optional[List[str]] = None
) -> List[schemas.SearchResult]:
    """Search for similar vectors with exclusion list"""
    query = db.query(Embeddings.element_id)
    
    if field == "image_embedding":
        query = query.order_by(Embeddings.image_embedding.cosine_distance(query_embedding))
    else:
        query = query.order_by(Embeddings.text_embedding.cosine_distance(query_embedding))
        
    if exclude_ids:
        query = query.filter(~Embeddings.element_id.in_(exclude_ids))
        
    results = query.offset(offset).limit(limit).all()
    
    return [schemas.SearchResult(image_id=str(r.element_id)) for r in results]
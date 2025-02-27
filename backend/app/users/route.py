import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import Optional, List

import requests

from ..config import get_env

from ..apiclient import APIClient

from ..database import get_db
from sqlalchemy.orm import Session
from .utils import validate_user
from . import utils, schemas, models, crud
import datetime
import uuid
import blurhash
from ..minioclient import minio_client
from ..pgcache import cached
from argon2 import PasswordHasher


 
router = APIRouter(prefix="/api/v1")
ph = PasswordHasher()


@router.post("/token", response_model=None, tags=["token"])
async def token(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = await crud.check_user_exists(db, username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    try:
        ph.verify(user.password, password)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password"
        )
    token = await utils.create_access_token(
        data={"user_id": str(user.id)}, 
        expires_delta=datetime.timedelta(days=30)
    )
    response = JSONResponse(content={"access_token": token, "token_type": "bearer"})
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=False,             # Allow JS to access the cookie
        secure=False,               # Set to True in production
        samesite="Lax",
        path="/",                    # Ensure the path is correct
    )
    return response

@router.post("/user", response_model=schemas.Id, tags=["users"])
async def create_user(
    name: str = Form(...),
    username: str = Form(...), 
    password: str = Form(...),
    bio: str = Form(None),
    avatar: UploadFile = File(None),
    db: Session = Depends(get_db),
):
    try:
        if await crud.check_user_exists(db, username):
            print("Username already exists")
            raise HTTPException(detail="Username already exists", status_code=400)
        password = ph.hash(password)
        if not avatar:
            avatarUrl = "https://www.gravatar.com/avatar/" + str(uuid.uuid4()) + "?d=identicon"
        else:
            thumbnail_binary = await utils.process_avatar(avatar)
            avatarUrl = await minio_client.upload(filename=avatar.filename, file_content=thumbnail_binary)
        user = models.Users(
            name=name,
            username=username,
            password=password,
            bio=bio,
            avatar=avatarUrl,
        )
        await crud.insert(db, user)
        await crud.init_clusters(db, user.id, user.username)
        return {"id": user.id}
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/user/{username}", response_model=schemas.UserDetail, tags=["users"])
# @cached(expire_in=300)
async def get_user_from_db(
    username: str, 
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user)
):
    user = None
    if username == "me" and user_id:
        print(user_id, type(user_id))
        user = await crud.get_user_by_id(db, user_id)
    else:
        user = await crud.get_user_by_username(db, username)

    print(user.name)
        
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    clusters = await crud.get_user_clusters(db, user.id, user.username)
    stats = await crud.get_user_stats(db, user.id)
    stats.clusters = len(clusters)
    print(stats)
    user = schemas.UserDetail(**user.__dict__, clusters=clusters, stats=stats)
    return user

@router.put("/user", response_model=schemas.Id, tags=["users"])
async def update_user(
    name: Optional[str] = Form(None),
    bio: Optional[str] = Form(None),
    avatar: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user)
):
    user = await crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    if avatar:
        avatarUrl = await minio_client.upload(await avatar.read())
        user.avatar = avatarUrl
    if name:
        user.name = name
    if bio:
        user.bio = bio
    await crud.update(db, user)
    return {"id": user.id}


@router.post("/cluster", response_model=schemas.Id, tags=["cluster"])
async def create_cluster(
    clusterForm: schemas.createClusterForm,
    db: Session = Depends(get_db),
    user_id:str = Depends(validate_user)
):
    user = await crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    cluster = await crud.get_cluster_by_title(db, clusterForm.title, user.id)
    if cluster:
        raise HTTPException(
            status_code=400,
            detail="Cluster already exists"
        )
    href = f"/{user.username}/{clusterForm.title}"
    cluster = models.Clusters(
        **clusterForm.__dict__,
        href=href,
        user_id=user_id
    )
    await crud.insert(db, cluster) 
    return {"id": cluster.id}   

@router.get("/user/{username}/cluster/{title}", response_model=schemas.ClusterDetail, tags=["cluster"])
@cached(expire_in=300)
async def get_user_cluster(
    username: str, 
    title: str, 
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user)
):
    user = await crud.get_user_by_username(db, username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    cluster = await crud.get_cluster_by_title(db, title, user.id)
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cluster not found"
        )
    elements = await crud.get_elements_in_cluster(db, cluster.id)
    is_following = True if user_id != str(cluster.user_id) else None
    stats = schemas.ClusterStats(elements=len(elements), followers=0, is_following=is_following)
    cluster = schemas.ClusterDetail(elements=elements, username=user.username, stats=stats, **cluster.__dict__)
    return cluster

# Helper functions for background tasks
async def create_element_index(
    element_id: str,
    user_id: str,
    contents: bytes,
    title: str,
    db: Session,
    version: int = 1
):
    try:
        # Generate tags
        tags = await APIClient.predict_tags(contents)
        for tag in tags:
            await crud.insert_tag(db, tag[0], tag[1], element_id)
        tags = await crud.get_element_tags(db, element_id)
        print(tags)
        
        # Generate embeddings
        image_embedding = await APIClient.img2vec(contents)
        text_embedding = await APIClient.text2vec(title)
        await crud.insert_embedding(db, element_id, text_embedding, image_embedding, version)
        
    except Exception as e:
        print(f"Error indexing element {element_id}: {str(e)}")

@router.post("/element", response_model=schemas.Id, tags=["element"])
async def create_element(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    desc: str = Form(None),
    image: UploadFile = File(...),
    source: Optional[str] = Form(None),
    cluster: str = Form(...),
    index: bool = Form(False),
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user),
):
    try:
        contents = await image.read()
        
        with Image.open(io.BytesIO(contents)) as img:
            img = img.convert("RGB")
            hash_img = img.copy()
            blur_img = img.copy()
            
            hash = await utils.hash(hash_img)
            dupl = await crud.get_element_by_hash(db, hash)
            
            filename = id = str(uuid.uuid4())
            url = await minio_client.upload(filename, contents) if not dupl else dupl.url
            placeholder = blurhash.encode(blur_img, x_components=4, y_components=3)
            
            cluster = await crud.get_or_create_cluster(db, cluster, user_id)
            source = await crud.get_or_insert_source(db, source)

            element = models.Elements(
                id=id,
                url=url,
                title=title,
                description=desc if desc else None,
                content_hash=hash,
                placeholder=placeholder,
                source_id=source.id,
                cluster_id=cluster.id,
                user_id=user_id
            )
            
            await crud.insert(db, element)
            
            if index:
                background_tasks.add_task(
                    create_element_index,
                    element_id=id,
                    user_id=user_id,
                    contents=contents,
                    title=title,
                    db=db
                )
            
            return {"id": str(element.id)}
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )



@router.post("/element/index", tags=["element"])
async def index_element(
    background_tasks: BackgroundTasks,
    element_id: str = Form(...),
    version: int = Form(get_env().embedding_version),
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user),
):
    try:
        element = await crud.get_element(db, element_id)
        if not element:
            raise HTTPException(status_code=404, detail="Element not found")
            
        image_binary = await utils.url_to_binary(element.url)
        
        background_tasks.add_task(
            create_element_index,
            element_id=element_id,
            user_id=user_id,
            contents=image_binary,
            title=element.title,
            db=db,
            version=version
        )
        
        return {"status": "Indexing started", "element_id": element_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def bulk_index_elements(
    element_ids: List[str],
    index_type: str,
    version: int,
    db: Session
):
    for element_id in element_ids:
        try:
            element = await crud.get_element(db, element_id)
            image_binary = await utils.url_to_binary(element.url)
            
            if index_type in ["embeddings", "all"]:
                image_embedding = await APIClient.img2vec(image_binary)
                text_embedding = await APIClient.text2vec(element.title)
                await crud.insert_embedding(db, element_id, text_embedding, image_embedding, version)
                
            if index_type in ["tags", "all"]:
                tags = await APIClient.predict_tags(image_binary)
                await crud.insert_tags(db, tags, element_id)
                
        except Exception as e:
            print(f"Error processing element {element_id}: {str(e)}")
            continue

@router.post("/element/index/all", tags=["element"])
async def index_all_elements(
    background_tasks: BackgroundTasks,
    version: str = Form("v1"),
    index_type: str = Form("embeddings"),  # embeddings, tags, or all
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user),
):
    try:
        # Get all element IDs
        element_ids = await crud.get_all_element_ids(db)
        
        background_tasks.add_task(
            utils.bulk_index_elements,
            element_ids=element_ids,
            index_type=index_type,
            version=version,
            db=db
        )
        
        return {
            "status": "Bulk indexing started",
            "total_elements": len(element_ids),
            "index_type": index_type,
            "version": version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/element/{element_id}", response_model=schemas.ElementPage, tags=["element"])
@cached(expire_in=300)
async def get_element(
    element_id: str, 
    user_id: str = Depends(validate_user),
    db: Session = Depends(get_db)
):
    element = await crud.get_element(db, element_id)
    if element is None:
        raise HTTPException(status_code=404, detail="Element Not Found")
    image_content = requests.get(element.url).content
    similar_element_ids = await crud.visual_search(db, image_content, APIClient.img2vec)
    similar_elements = [await crud.get_element(db, id) for id, _ in similar_element_ids]
    return schemas.ElementPage(**element.__dict__, similar=similar_elements)

@router.post("/search", response_model=schemas.ElementList, tags=["search"])
@cached(expire_in=300)
async def search(
    query: str = Form(...),
    user_id: str = Depends(validate_user),
    db: Session = Depends(get_db)
):
    try:
        # Convert query to embedding and search
        query_embedding = await APIClient.text2vec(query)
        results = await crud.vector_search(
            db=db,
            query_embedding=query_embedding,
            field="text_embedding"
        )
        
        # Get full element details
        elements = []
        for result in results:
            element = await crud.get_element(db, result.image_id)
            if element:
                elements.append(element)
                
        return schemas.ElementList(elements=elements)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/visual-search", response_model=schemas.SimilarElements, tags=["search"])
@cached(expire_in=300)
async def visual_search(
    id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user)
):
    try:
        element, query_embedding = None, None

        if id:
            # Get element and its embedding
            element = await crud.get_element(db, id)
            if not element:
                raise HTTPException(status_code=404, detail="Element not found")
            
            embedding = await crud.get_element_embedding(db, id)
            if not embedding:
                raise HTTPException(status_code=404, detail="Embedding not found")
            query_embedding = embedding.image_embedding
            
        else:
            # Process uploaded image
            if not image:
                raise HTTPException(status_code=400, detail="No image provided")
                
            contents = await image.read()
            with Image.open(io.BytesIO(contents)) as img:
                img = img.convert("RGB")
                query_embedding = await APIClient.img2vec(img)
            
            element = schemas.ElementBase(
                url=None, title=None, description=None,
                content_hash=None, placeholder=None,
                source_id=None, cluster_id=None, user_id=None
            )

        # Perform vector search
        results = await crud.vector_search(
            db=db,
            query_embedding=query_embedding,
            field="image_embedding"
        )
        
        # Get full element details
        similar_elements = []
        for result in results:
            similar = await crud.get_element(db, result.image_id)
            if similar:
                similar_elements.append(similar)

        return schemas.SimilarElements(
            element=element,
            similar=similar_elements
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )
    
@router.get("/feed", tags=["curation"], response_model=schemas.ElementList)
async def get_recommendations(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user)
):
    try:
        # Get user's liked and interacted elements
        liked_elements = await crud.get_user_likes(db, user_id)
        interacted_elements = await crud.get_user_interactions(db, user_id)
        
        if not liked_elements and not interacted_elements:
            # If no interactions, return popular/trending elements
            elements = await crud.get_trending_elements(db, limit, randomize=True)
            return schemas.ElementList(elements=elements)
            
        # Get embeddings for liked/interacted elements
        embeddings:List[models.Embeddings] = []
        for element_id in set(liked_elements + interacted_elements):
            embedding = await crud.get_element_embedding(db, element_id)
            if embedding:
                embeddings.append(embedding)
                
        if not embeddings:
            elements = await crud.get_trending_elements(db, limit, randomize=True)
            return schemas.ElementList(elements=elements)
            
        # Average the embeddings to get user interest vector
        image_embeddings = [e.image_embedding for e in embeddings]
        text_embeddings = [e.text_embedding for e in embeddings]
        
        avg_image_embedding = np.mean(image_embeddings, axis=0)
        avg_text_embedding = np.mean(text_embeddings, axis=0)
        
        # Search for similar elements using both embeddings
        image_results = await crud.vector_search(
            db=db,
            query_embedding=avg_image_embedding.tolist(),
            field="image_embedding",
            limit=limit,
            offset=offset,
            exclude_ids=liked_elements + interacted_elements  # Exclude already seen elements
        )
        
        text_results = await crud.vector_search(
            db=db,
            query_embedding=avg_text_embedding.tolist(),
            field="text_embedding",
            limit=limit,
            offset=offset,
            exclude_ids=liked_elements + interacted_elements
        )
        
        # Combine and deduplicate results
        result_ids = {r.image_id for r in image_results + text_results}
        elements = []
        for element_id in result_ids:
            element = await crud.get_element(db, element_id)
            if element:
                elements.append(element)
                
        return schemas.ElementList(elements=elements[:limit])
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/explore", tags=["curation"], response_model=schemas.ClusterList)
@cached(expire_in=300)
async def explore(user_id: str = Depends(validate_user)):
    pass


@router.post("/interactions", response_model=schemas.UserInteractionOut)
async def create_interaction(interaction_in: schemas.UserInteractionCreate, db: Session = Depends(get_db)):
    try:
        interaction = await crud.create_user_interaction(
            db,
            user_id=str(interaction_in.user_id),
            element_id=str(interaction_in.element_id),
            interaction_type=schemas.InteractionTypeEnum(interaction_in.interaction_type),
            time_spent_seconds=interaction_in.time_spent_seconds
        )
        return interaction
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
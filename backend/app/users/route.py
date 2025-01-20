import random
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, BackgroundTasks
from PIL import Image
import io
from typing import Optional

import numpy as np
from ..database import get_db
from sqlalchemy.orm import Session
from .utils import validate_user
from . import utils, schemas, models, crud
import datetime
import uuid
import blurhash
from ..inference import manager, ClipEmbedder
from ..minio import minio_client
from ..milvus import milvus_service

router = APIRouter(prefix="/api/v1")

from fastapi.responses import JSONResponse

@router.post("/token", response_model=None, tags=["token"])
async def dummy_token(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = await crud.get_user_by_username(db, username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    if not await utils.verify_password(password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password"
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
        if await crud.get_user_by_username(db, username):
            print("Username already exists")
            raise HTTPException(detail="Username already exists", status_code=400)
        password = await utils.encrypt(password)
        if not avatar:
            avatarUrl = "https://www.gravatar.com/avatar/" + str(uuid.uuid4()) + "?d=identicon"
        else:
            avatarUrl = await minio_client.upload(await avatar.read())
        user = models.Users(
            name=name,
            username=username,
            password=password,
            bio=bio,
            avatar=avatarUrl,
        )
        await crud.insert(db, user)
    # await crud.init_clusters(db, kwargs)
        return {"id": user.id}
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/user/{username}", response_model=schemas.UserDetail, tags=["users"])  
async def get_user_from_db(
    username: str, 
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user)
):
    user = None
    if username == "me" and user_id:
        user = await crud.get_user_by_id(db, user_id)
    else:
        user = await crud.get_user_by_username(db, username, user_id)
        
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    clusters = await crud.get_user_clusters(db, user.id, user.username)
    stats = await crud.get_user_stats(db, user.id)
    stats.clusters = len(clusters)
    user = schemas.UserDetail(**user.__dict__, clusters=clusters, stats=stats)
    return user

@router.put("/user/{username}", response_model=schemas.Id, tags=["users"])
async def update_user(
    username: str,
    name: str = Form(None),
    bio: str = Form(None),
    avatar: UploadFile = File(None),
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
    elements = await crud.get_elements_in_cluster(db, cluster.id, user.id)
    is_following = True if user_id != str(cluster.user_id) else None
    stats = schemas.ClusterStats(elements=len(elements), followers=0, is_following=is_following)
    cluster = schemas.ClusterDetail(elements=elements, username=user.username, stats=stats, **cluster.__dict__)
    return cluster


@router.post("/element", response_model=schemas.Id, tags=["element"])
async def create_element(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    desc: str = Form(None),
    image: UploadFile = File(...),
    source: Optional[str] = Form(None),
    cluster: str = Form(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user),
):
    try:
        contents = await image.read()
        
        # Use context manager for image handling
        with Image.open(io.BytesIO(contents)) as img:
            img = img.convert("RGB")
            
            # Create separate copies for each operation
            embed_img = img.copy()
            hash_img = img.copy()
            blur_img = img.copy()
            
            # Generate embeddings
            async with manager.get_model("clip", ClipEmbedder) as model:
                result = await model.img2vec(embed_img)
            if result is None:
                raise ValueError("Failed to generate embeddings")
                
            # Generate hash and check duplicates
            hash = await utils.hash(hash_img)
            dupl = await crud.get_element_by_hash(db, hash)
            
            # Upload and generate placeholder
            url = await minio_client.upload(contents) if not dupl else dupl.url
            placeholder = blurhash.encode(blur_img, x_components=4, y_components=3)
            
            # Get cluster
            cluster = await crud.get_cluster_by_title(db, cluster, user_id)
            if not cluster:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Cluster not found"
                )

            # Create element
            element = models.Elements(
                url=url,
                title=title,
                desc=desc if desc else None,
                hash=hash,
                placeholder=placeholder,
                analysis=None,
                source=source,
                cluster_id=cluster.id,
                user_id=user_id
            )
            
            await crud.insert(db, element)
            
            # Pass fresh image copy to background task
            process_img = img.copy()
            background_tasks.add_task(
                utils.process_embeddings,
                str(element.id),
                process_img,
                title + desc if desc else title
            )
            
            return {"id": str(element.id)}
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/element/{id}", response_model=schemas.ElementPage, tags=["element"])
async def get_element(
    id: str, 
    user_id: str = Depends(validate_user),
    db: Session = Depends(get_db)
):
    element = await crud.get_element(db, id)
    if element is None:
        raise HTTPException(status_code=404, detail="Element Not Found")
    img_embed = milvus_service.get(id)['image_embedding']
    results = milvus_service.search(img_embed, field="image_embedding")
    similar = [await crud.get_element(db, result.image_id) for result in results]
    return schemas.ElementPage(**element.__dict__, similar=similar)

@router.post("/search", response_model=None, tags=["search"])
async def search(
    query: str = Form(...),
    user_id: str = Depends(validate_user),
    db: Session = Depends(get_db)
    ):
    try:
        async with manager.get_model("clip", ClipEmbedder) as model:
            query_embedding = await model.text2vec(query)
        results = milvus_service.search(
            query_embedding=query_embedding,
            field="text_embedding"
        )
        elements = [await crud.get_element(db, result.image_id) for result in results]
        return schemas.ElementList(elements=elements)
    except Exception as e:
        return {"error": str(e)}

@router.post("/visual-search", response_model=schemas.SimilarElements, tags=["search"])
async def visual_search(
    id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user)):
    try:      
        element, img_embed = None, None  
        if id:        
            element = await crud.get_element(db, id)
            img_embed = milvus_service.get(id)['image_embedding']
        else:
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            async with manager.get_model("clip", ClipEmbedder) as model:
                img_embed = await model.img2vec(pil_image)
            element = schemas.ElementBase(url=None, title=None, desc=None, hash=None, placeholder=None, analysis=None, source=None, cluster_id=None, user_id=None)
        results = milvus_service.search(
            img_embed,
            field="image_embedding"
        )
        elements = [await crud.get_element(db, result.image_id) for result in results]
        return schemas.SimilarElements(
            element=element,
            similar=elements
        )
    except Exception as e:
        return {"error": str(e)}

# @router.post("/visual-search", response_model=schemas.ElementList, tags=["search"])
# async def visual_search(
#     id: Optional[str] = Form(None),
#     db: Session = Depends(get_db),
#     user_id: str = Depends(validate_user)):
#     try:
#         contents = await image.read()
#         pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
#         async with manager.get_model("clip", ClipEmbedder) as model:
#             img_embed = await model.img2vec(pil_image)
                

#         elements = [await crud.get_element(db, result.image_id) for result in results]
#         return schemas.ElementList(elements=elements)
#     except Exception as e:
#         return {"error": str(e)}

@router.get("/feed", tags=["curation"], response_model=schemas.Recommedations)
async def get_recommendations(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user)
):
    engine = crud.RecommendationEngine(db, milvus_service)
    recommendations = await engine.get_recommendations(
        user_id=user_id,
        limit=limit,
        offset=offset
    )
    return schemas.Recommedations(elements=recommendations)

@router.get("/explore", tags=["curation"], response_model=schemas.ClusterList)
async def explore(user_id: str = Depends(validate_user)):
    pass

# future merge wwithhh put
@router.post("/likes/{image_id}", tags=["likes"])
async def like_image(
    image_id: str,
    user_id: str = Depends(validate_user),
    db: Session = Depends(get_db)
):
    await crud.add_user_like(db, user_id, image_id)
    return {"status": "success"}
from pydantic import BaseModel, UUID4
from typing import List, Optional, Union

# Note add thumbnail to clusters and stats followers, elements, owner to clusters, is followed, stats cluster, elements, likes, element stats is liked

class Token(BaseModel):
    token: str
    token_type: str

class AuthForm(BaseModel):
    username: str
    password: str

class createUserForm(BaseModel):
    name: str
    username: str
    password: str
    about: str
    avatar_image: str
    tags: str

class createClusterForm(BaseModel):
    title: str
    desc: str

class ClusterBase(BaseModel):
    title: str
    href: str
    thumbnail: Optional[str] = None
    count: Optional[int] = None


    class Config:
        extra = "ignore"

class UserBase(BaseModel):
    name: str
    username: str
    bio: Optional[str] = None
    avatar: Optional[str] = None

    class Config:
        from_attributes = True
        extra = "ignore"

class UserStats(BaseModel):
    clusters: int
    elements: int
    likes: int

    class Config:
        from_attributes = True
        extra = "ignore"

class UserDetail(UserBase):
    stats: UserStats
    clusters: List[ClusterBase]

    class Config:
        from_attributes = True
        extra = "ignore"

class ElementBase(BaseModel):
    id: str
    url: str
    title: str
    desc: Optional[str] = None
    placeholder: Optional[str] = None
    source: Optional[str] = None

    class Config:
        from_attributes = True
        extra = "ignore"
        coerce_numbers_to_str = True

class ElementPage(ElementBase):
    similar: List[ElementBase] = []

class ClusterStats(BaseModel):
    elements: int
    followers: int
    is_following: bool|None

    class Config:
        from_attributes = True
        extra = "ignore"

class ClusterDetail(ClusterBase):
    desc: str
    username: str
    avatar: Optional[str] = None
    stats: ClusterStats
    elements: List[ElementBase]

    class Config:
        from_attributes = True
        extra = "ignore"

class ElementDetail(ElementBase):
    id: int

    class Config:
        from_attributes = True
        extra = "ignore"

class ElementList(BaseModel):
    elements: List[ElementBase]

    class Config:
        from_attributes = True
        extra = "ignore"

class Recommedations(ElementList):
    Tags: List[str] = ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10"]
    class Config:
        from_attributes = True
        extra = "ignore"

class ClusterList(BaseModel):
    clusters: List[ClusterDetail]

    class Config:
        from_attributes = True
        extra = "ignore"

class Id(BaseModel):
    # id might be uuid or int
    id: Union[UUID4, str]

    class Config:
        from_attributes = True
        extra = "ignore"

# change to vector results
class SearchResult(BaseModel):
    image_id: str
    score: float

class SimilarElements(ElementBase):
    similar: List[ElementBase]

    class Config:
        from_attributes = True
        extra = "ignore"
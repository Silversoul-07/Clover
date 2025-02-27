from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/publicdb"

    minio_url: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "mybucket"
    storage_path: str = "storage"

    jwt_secret_key: str = '''e2cf6c7e8196773130676cbddac66d46f49e08c6d07958805413d7d1c8ac4e310ff4dcbfd2f3d65076a3f74ad1d941f8ea51acb19a7e66b237bb76044460bc85e7ad6ab9177972065d8e02bb563b6dcba102f677855b04b6ec621158f4ae329b066f89d8f7cf0f72ff1497576cd0002e7df89969181b3ac10b398628041c79c5ffab5ce79151e5d7ccbce23a54493e14982dcfb7ce13fa849b12f63f1800f8d2d4a2061e424852e58f0961db043905f3be6a22000f71e2163e398d88178dbbf60b4e8c912341f9dd4dee8efdc2c63ebb4f4c330c25c4a91453408c3a3329bf56cad8c37614cb1b60b9aaf92b109c3a1a5adb2f8abeb472abc36a695bbbdc3a3a'''

    # attribute for text and image embedding size
    txtEmdSize: int = 1024
    imgEmdSize: int = 1024
    embedding_version: int = 1

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_env():
    return Settings()  

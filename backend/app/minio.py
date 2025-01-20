import os
import json
import io
from uuid import uuid4
from typing import Optional
from minio import Minio
from PIL import Image

class MinioClient:
    def __init__(self, bucket_name: str) -> None:
        minio_endpoint = 'minio:9000'
        minio_access_key = 'minioadmin'
        minio_secret_key = 'minioadmin'
        self.client = Minio(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        self.bucket_name = bucket_name

    def set_bucket_policy(self, bucket_name: str) -> None:
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
                }
            ]
        }
        policy_json = json.dumps(policy)
        self.client.set_bucket_policy(bucket_name, policy_json)

    def create_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)
            self.set_bucket_policy(self.bucket_name)

    async def upload(
        self,
        file_content: bytes,
        content_type: Optional[str] = None
    ) -> Optional[str]:
        try:
            filename = str(uuid4())
            file_like_object = io.BytesIO(file_content)
            
            # Detect file type using magic numbers if content_type not provided
            if content_type is None:
                import magic
                detected_type = magic.from_buffer(file_content, mime=True)
                content_type = detected_type
                if not content_type:
                    raise Exception("Could not detect file type")
                
                # # Fallback to extension-based detection if magic fails
                # if content_type == 'application/octet-stream':
                #     if filename.lower().endswith((
                #         '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'
                #     )):
                #         content_type = 'video/mp4'
                #     elif filename.lower().endswith((
                #         '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'
                #     )):
                #         content_type = 'image/jpeg'
                #     elif filename.lower().endswith((
                #         '.mp3', '.wav', '.flac', '.aac', '.ogg'
                #     )):
                #         content_type = 'audio/mpeg'

            self.client.put_object(
                self.bucket_name,
                filename,
                file_like_object,
                length=len(file_content),
                content_type=content_type
            )
            return f"http://localhost:9000/{self.bucket_name}/{filename}"

        except Exception as e:
            print(f"Error uploading to MinIO: {e}")
            return None
        
minio_client = MinioClient("media")
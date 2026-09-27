from minio import Minio

from knowledge.core.settings import Settings


def create_minio_client(settings: Settings) -> Minio:
    return Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key.get_secret_value(),
        secure=settings.minio_secure,
    )


def ensure_minio_bucket(client: Minio, bucket_name: str) -> bool:
    if client.bucket_exists(bucket_name):
        return False

    client.make_bucket(bucket_name)
    return True

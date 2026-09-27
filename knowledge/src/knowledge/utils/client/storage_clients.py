from io import BytesIO

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


def upload_text(
    client: Minio,
    bucket_name: str,
    object_name: str,
    content: str,
) -> None:
    data = content.encode("utf-8")

    client.put_object(
        bucket_name=bucket_name,
        object_name=object_name,
        data=BytesIO(data),
        length=len(data),
        content_type="text/plain; charset=utf-8",
    )


def download_text(
    client: Minio,
    bucket_name: str,
    object_name: str,
) -> str:
    response = client.get_object(bucket_name, object_name)

    try:
        return response.read().decode("utf-8")
    finally:
        response.close()
        response.release_conn()

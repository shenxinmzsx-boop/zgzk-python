from io import BytesIO
from typing import Any

from minio import Minio
from pymongo import MongoClient

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


def upload_bytes(
    client: Minio,
    bucket_name: str,
    object_name: str,
    data: bytes,
    content_type: str,
) -> None:
    client.put_object(
        bucket_name=bucket_name,
        object_name=object_name,
        data=BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


def upload_text(
    client: Minio,
    bucket_name: str,
    object_name: str,
    content: str,
) -> None:
    data = content.encode("utf-8")

    upload_bytes(
        client=client,
        bucket_name=bucket_name,
        object_name=object_name,
        data=data,
        content_type="text/plain; charset=utf-8",
    )


def download_bytes(
    client: Minio,
    bucket_name: str,
    object_name: str,
) -> bytes:
    response = client.get_object(bucket_name, object_name)

    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def download_text(
    client: Minio,
    bucket_name: str,
    object_name: str,
) -> str:
    data = download_bytes(
        client=client,
        bucket_name=bucket_name,
        object_name=object_name,
    )

    return data.decode("utf-8")


def create_mongo_client(settings: Settings) -> MongoClient[dict[str, Any]]:
    return MongoClient(
        host=settings.mongo_host,
        port=settings.mongo_port,
        username=settings.mongo_username,
        password=settings.mongo_password.get_secret_value(),
        authSource=settings.mongo_auth_source,
        serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
    )


def ping_mongodb(client: MongoClient[dict[str, Any]]) -> None:
    client.admin.command("ping")

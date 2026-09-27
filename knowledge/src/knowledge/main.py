from knowledge.core.settings import get_settings
from knowledge.utils.client.storage_clients import (
    create_minio_client,
    ensure_minio_bucket,
)


def main() -> None:
    settings = get_settings()
    client = create_minio_client(settings)

    created = ensure_minio_bucket(
        client=client,
        bucket_name=settings.minio_bucket_name,
    )

    status = "已创建" if created else "已存在"

    print("MinIO 连接成功")
    print(f"Bucket {status}: {settings.minio_bucket_name}")


if __name__ == "__main__":
    main()

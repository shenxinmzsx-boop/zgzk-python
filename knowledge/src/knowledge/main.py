from hashlib import sha256
from pathlib import Path

from knowledge.core.settings import PROJECT_ROOT, get_settings
from knowledge.utils.client.storage_clients import (
    create_minio_client,
    create_mongo_client,
    download_bytes,
    ensure_minio_bucket,
    ping_mongodb,
    upload_bytes,
)
from knowledge.utils.document_metadata import (
    ensure_document_metadata_indexes,
    find_document_metadata,
    upsert_document_metadata,
)


def read_file_with_sha256(file_path: Path) -> tuple[bytes, str]:
    content = file_path.read_bytes()
    digest = sha256(content).hexdigest()
    return content, digest


def main() -> None:
    settings = get_settings()
    client = create_minio_client(settings)

    bucket_created = ensure_minio_bucket(
        client=client,
        bucket_name=settings.minio_bucket_name,
    )

    status = "已创建" if bucket_created else "已存在"

    print("MinIO 连接成功")
    print(f"Bucket {status}: {settings.minio_bucket_name}")

    source_path = PROJECT_ROOT / "examples" / "sample_knowledge.txt"
    object_name = "documents/sample_knowledge.txt"
    content_type = "text/plain; charset=utf-8"

    source_data, source_sha256 = read_file_with_sha256(source_path)

    upload_bytes(
        client=client,
        bucket_name=settings.minio_bucket_name,
        object_name=object_name,
        data=source_data,
        content_type=content_type,
    )

    downloaded_data = download_bytes(
        client=client,
        bucket_name=settings.minio_bucket_name,
        object_name=object_name,
    )

    downloaded_sha256 = sha256(downloaded_data).hexdigest()

    if downloaded_sha256 != source_sha256:
        raise RuntimeError("MinIO 下载内容与本地文件 SHA-256 不一致")

    mongo_client = create_mongo_client(settings)

    try:
        ping_mongodb(mongo_client)

        collection = mongo_client[
            settings.mongo_database
        ][
            settings.mongo_documents_collection
        ]

        index_name = ensure_document_metadata_indexes(collection)
        print(f"MongoDB 索引已就绪: {index_name}")
        metadata_created = upsert_document_metadata(
            collection,
            bucket_name=settings.minio_bucket_name,
            object_name=object_name,
            original_filename=source_path.name,
            content_type=content_type,
            size=len(source_data),
            sha256=source_sha256,
        )

        metadata = find_document_metadata(
            collection,
            bucket_name=settings.minio_bucket_name,
            object_name=object_name,
        )

        if metadata is None:
            raise RuntimeError("MongoDB 未查询到刚写入的对象元数据")

        metadata_status = "已创建" if metadata_created else "已存在或已更新"

        print("MongoDB 连接成功")
        print(f"元数据{metadata_status}: {metadata}")
    finally:
        mongo_client.close()

    print(f"对象读写成功: {object_name}")
    print(f"原始文件: {source_path.name}")
    print(f"文件大小: {len(source_data)} bytes")
    print(f"SHA-256: {source_sha256}")


if __name__ == "__main__":
    main()

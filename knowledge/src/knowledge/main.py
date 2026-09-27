from knowledge.core.settings import get_settings
from knowledge.utils.client.storage_clients import (
    create_minio_client,
    create_mongo_client,
    download_text,
    ensure_minio_bucket,
    ping_mongodb,
    upload_text,
)
from knowledge.utils.document_metadata import (
    find_document_metadata,
    upsert_document_metadata,
)


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

    object_name = "smoke-test/hello.txt"
    expected_content = "掌柜智库MinIO读写测试"

    upload_text(
        client=client,
        bucket_name=settings.minio_bucket_name,
        object_name=object_name,
        content=expected_content,
    )

    actual_content = download_text(
        client=client,
        bucket_name=settings.minio_bucket_name,
        object_name=object_name,
    )

    if actual_content != expected_content:
        raise RuntimeError("MinIO 下载内容与上传内容不一致")

    mongo_client = create_mongo_client(settings)

    try:
        ping_mongodb(mongo_client)

        collection = mongo_client[
            settings.mongo_database
        ][
            settings.mongo_documents_collection
        ]

        metadata_created = upsert_document_metadata(
            collection,
            bucket_name=settings.minio_bucket_name,
            object_name=object_name,
            content_type="text/plain; charset=utf-8",
            size=len(expected_content.encode("utf-8")),
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
    print(f"对象内容: {actual_content}")


if __name__ == "__main__":
    main()

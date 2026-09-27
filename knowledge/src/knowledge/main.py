from knowledge.core.settings import get_settings
from knowledge.utils.client.storage_clients import (
    create_minio_client,
    download_text,
    ensure_minio_bucket,
    upload_text,
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

    print(f"对象读写成功: {object_name}")
    print(f"对象内容: {actual_content}")


if __name__ == "__main__":
    main()

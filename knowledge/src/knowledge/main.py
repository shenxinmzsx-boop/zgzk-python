from knowledge.core.settings import get_settings

def main() -> None:
    settings = get_settings()

    print("掌柜智库启动成功")
    print(f"Milvus: {settings.milvus_uri}")
    print(f"MongoDB: {settings.mongo_host}:{settings.mongo_port}")
    print(f"MongoDB password: {settings.mongo_password}")
    print(f"MinIO: {settings.minio_endpoint}")
    print(f"MinIO secret: {settings.minio_secret_key}")


if __name__ == "__main__":
    main()
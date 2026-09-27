# 掌柜智库 Docker 服务

## 使用方法

1. 将 `.env.example` 复制为 `.env`。
2. 把 `.env` 中所有 `CHANGE_ME` 替换为自己的强密码。
3. 在当前目录执行：

```powershell
docker compose config
docker compose pull
docker compose up -d
docker compose ps
```

Milvus 首次启动可能需要几分钟。

## 本机访问地址

- Attu: http://localhost:7000
- Neo4j: http://localhost:7474
- MinIO: http://localhost:9001
- Milvus: http://localhost:19530
- Milvus 健康接口: http://localhost:9091/healthz
- MongoDB: `mongodb://localhost:27017`

## 项目 `.env` 连接配置

```ini
MILVUS_URL=http://localhost:19530
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=<与 Docker .env 中 MINIO_ROOT_USER 相同>
MINIO_SECRET_KEY=<与 Docker .env 中 MINIO_ROOT_PASSWORD 相同>
NEO4J_URI=bolt://localhost:7687
NEO4J_DATABASE=neo4j
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<与 Docker .env 中 NEO4J_PASSWORD 相同>
MONGO_URL=mongodb://admin:<MongoDB密码>@localhost:27017/?authSource=admin
```

## 常用命令

```powershell
docker compose ps
docker compose logs -f --tail=100
docker compose restart
docker compose down
docker compose up -d
```

不要提交 `.env`，其中包含密码。

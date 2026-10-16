from collections.abc import Mapping, Sequence
from hashlib import sha256
from json import dumps
from typing import Any

from numpy.ma.core import minimum
from pymilvus import CollectionSchema, DataType, MilvusClient
from pymilvus.client import search_result


def build_knowledge_chunks_schema(
    *,
    embedding_dimension: int,
) -> CollectionSchema:
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be > 0")

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=64,
    )
    schema.add_field(
        field_name="bucket_name",
        datatype=DataType.VARCHAR,
        max_length=63,
    )
    schema.add_field(
        field_name="object_name",
        datatype=DataType.VARCHAR,
        max_length=1024,
    )
    schema.add_field(
        field_name="document_sha256",
        datatype=DataType.VARCHAR,
        max_length=64,
    )
    schema.add_field(
        field_name="chunk_index",
        datatype=DataType.INT64,
    )
    schema.add_field(
        field_name="chunk_sha256",
        datatype=DataType.VARCHAR,
        max_length=64,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65_535,
    )
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=embedding_dimension,
    )

    return schema


def ensure_knowledge_chunks_collection(
    client: MilvusClient,
    *,
    collection_name: str,
    embedding_dimension: int,
) -> bool:
    if client.has_collection(collection_name=collection_name):
        return False

    schema = build_knowledge_chunks_schema(
        embedding_dimension=embedding_dimension,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )

    return True


def build_chunk_id(
    *,
    bucket_name: str,
    object_name: str,
    chunk_index: int,
) -> str:
    if not bucket_name.strip():
        raise ValueError("bucket_name must not be empty")

    if not object_name.strip():
        raise ValueError("object_name must not be empty")

    if chunk_index < 0:
        raise ValueError("chunk_index must be >= 0")

    identity = f"{bucket_name}\0{object_name}\0{chunk_index}"

    return sha256(identity.encode("utf-8")).hexdigest()


def upsert_knowledge_chunks(
    client: MilvusClient,
    *,
    collection_name: str,
    chunk_documents: Sequence[Mapping[str, Any]],
    embeddings: Sequence[Sequence[float]],
) -> list[str]:
    if not collection_name.strip():
        raise ValueError("collection_name must not be empty")

    if len(chunk_documents) != len(embeddings):
        raise ValueError("chunk_documents must have same length as embeddings")

    if not chunk_documents:
        return []

    entities: list[dict[str, Any]] = []
    chunk_ids: list[str] = []

    for document, embedding in zip(
        chunk_documents, embeddings, strict=True
    ):
        chunk_id = build_chunk_id(
            bucket_name=document["bucket_name"],
            object_name=document["object_name"],
            chunk_index=document["chunk_index"],
        )

        chunk_ids.append(chunk_id)
        entities.append(
            {
                "chunk_id": chunk_id,
                "bucket_name": document["bucket_name"],
                "object_name": document["object_name"],
                "document_sha256": document["document_sha256"],
                "chunk_index": document["chunk_index"],
                "chunk_sha256": document["chunk_sha256"],
                "text": document["text"],
                "embedding": list(embedding),
            }
        )

    result = client.upsert(
        collection_name=collection_name,
        data=entities,
    )

    if result.get("upsert_count") != len(entities):
        raise RuntimeError("Milvus upsert数量与提交数量不一致")

    return chunk_ids

def search_knowledge_chunks(
    client: MilvusClient,
    *,
    collection_name: str,
    query_embedding: Sequence[float],
    expected_dimension: int,
    limit: int = 3,
    minimum_similarity: float = 0.5,
) -> list[dict[str, Any]]:
    if not collection_name.strip():
        raise ValueError("collection_name must not be empty")

    if expected_dimension <= 0:
        raise ValueError("expected_dimension must be > 0")

    if len(query_embedding) != expected_dimension:
        raise ValueError(f"query_embedding dimension must be {expected_dimension}")

    if limit <= 0:
        raise ValueError("limit must be > 0")

    if not -1.0 <= minimum_similarity <= 1.0:
        raise ValueError("minimum_similarity must be between -1 and 1")

    search_results = client.search(
        collection_name=collection_name,
        data=[list(query_embedding)],
        anns_field="embedding",
        search_params={
            "metric_type": "COSINE",
        },
        limit=limit,
        output_fields=[
            "chunk_id",
            "bucket_name",
            "object_name",
            "chunk_index",
            "text"
        ],
        consistency_level="Strong",
    )

    if len(search_results) != 1:
        raise RuntimeError("Milvus 返回的查询结果组数量不正确")

    results = list(search_results[0])

    if any("distance" not in result for result in results):
        raise RuntimeError("Milvus 搜索结果缺少 distance")

    return [
        result
        for result in results
        if float(result["distance"]) >= minimum_similarity
    ]


def delete_stale_knowledge_chunks(
    client: MilvusClient,
    *,
    collection_name: str,
    bucket_name: str,
    object_name: str,
    first_stale_chunk_index: int,
) -> None:
    if not collection_name.strip():
        raise ValueError("collection_name must not be empty")

    if not bucket_name.strip():
        raise ValueError("bucket_name must not be empty")

    if not object_name.strip():
        raise ValueError("object_name must not be empty")

    if first_stale_chunk_index < 0:
        raise ValueError("first_stale_chunk_index must be >= 0")

    client.delete(
        collection_name=collection_name,
        filter=(
            f"bucket_name == {dumps(bucket_name)} and "
            f"object_name == {dumps(object_name)} and "
            f"chunk_index >= {first_stale_chunk_index}"
        ),
    )

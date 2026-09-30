from hashlib import sha256
from typing import Any

from pymongo import ASCENDING, UpdateOne
from pymongo.collection import Collection


def ensure_document_chunk_indexes(
    collection: Collection[dict[str, Any]],
) -> str:
    return collection.create_index(
        [
            ("bucket_name", ASCENDING),
            ("object_name", ASCENDING),
            ("chunk_index", ASCENDING),
        ],
        unique=True,
        name="uq_document_chunks_bucket_object_index",
    )


def sync_document_chunks(
    collection: Collection[dict[str, Any]],
    *,
    bucket_name: str,
    object_name: str,
    document_sha256: str,
    chunks: list[str],
) -> int:
    operations: list[UpdateOne] = []

    for chunk_index, text in enumerate(chunks):
        chunk_sha256 = sha256(text.encode("utf-8")).hexdigest()

        operations.append(
            UpdateOne(
                {
                    "bucket_name": bucket_name,
                    "object_name": object_name,
                    "chunk_index": chunk_index,
                },
                {
                    "$set": {
                        "bucket_name": bucket_name,
                        "object_name": object_name,
                        "document_sha256": document_sha256,
                        "chunk_index": chunk_index,
                        "text": text,
                        "char_count": len(text),
                        "chunk_sha256": chunk_sha256,
                    },
                },
                upsert=True,
            ),
        )

    if operations:
        collection.bulk_write(operations, ordered=True)

    collection.delete_many(
        {
            "bucket_name": bucket_name,
            "object_name": object_name,
            "chunk_index": {"$gte": len(chunks)},
        },
    )

    return len(chunks)

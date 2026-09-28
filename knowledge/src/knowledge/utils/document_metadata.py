from typing import Any

from pymongo import ASCENDING
from pymongo.collection import Collection


def upsert_document_metadata(
    collection: Collection[dict[str, Any]],
    *,
    bucket_name: str,
    object_name: str,
    original_filename: str,
    content_type: str,
    size: int,
    sha256: str,
) -> bool:
    result = collection.update_one(
        {
            "bucket_name": bucket_name,
            "object_name": object_name,
        },
        {
            "$set": {
                "original_filename": original_filename,
                "content_type": content_type,
                "size": size,
                "sha256": sha256,
            },
        },
        upsert=True,
    )

    return result.upserted_id is not None


def find_document_metadata(
    collection: Collection[dict[str, Any]],
    *,
    bucket_name: str,
    object_name: str,
) -> dict[str, Any] | None:
    return collection.find_one(
        {
            "bucket_name": bucket_name,
            "object_name": object_name,
        },
        {"_id": False},
    )


def ensure_document_metadata_indexes(
    collection: Collection[dict[str, Any]],
) -> str:
    return collection.create_index(
        [
            ("bucket_name", ASCENDING),
            ("object_name", ASCENDING),
        ],
        unique=True,
        name="uq_documents_bucket_object",
    )

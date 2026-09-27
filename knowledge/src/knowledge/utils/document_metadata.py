from typing import Any

from pymongo.collection import Collection


def upsert_document_metadata(
    collection: Collection[dict[str, Any]],
    *,
    bucket_name: str,
    object_name: str,
    content_type: str,
    size: int,
) -> bool:
    result = collection.update_one(
        {
            "bucket_name": bucket_name,
            "object_name": object_name,
        },
        {
            "$set": {
                "content_type": content_type,
                "size": size,
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

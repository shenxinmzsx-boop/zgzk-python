import unittest
from unittest.mock import MagicMock

from pymilvus import MilvusClient

from knowledge.utils.vector_store import (
    build_chunk_id,
    build_knowledge_chunks_schema,
    delete_stale_knowledge_chunks,
    ensure_knowledge_chunks_collection,
    search_knowledge_chunks,
    upsert_knowledge_chunks,
)


class BuildChunkIdTests(unittest.TestCase):
    def test_returns_stable_sha256_hex_string(self) -> None:
        first = build_chunk_id(
            bucket_name="knowledge-base",
            object_name="documents/sample.txt",
            chunk_index=0,
        )
        second = build_chunk_id(
            bucket_name="knowledge-base",
            object_name="documents/sample.txt",
            chunk_index=0,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in first))

    def test_separates_different_objects(self) -> None:
        first = build_chunk_id(
            bucket_name="knowledge-base",
            object_name="documents/first.txt",
            chunk_index=0,
        )
        second = build_chunk_id(
            bucket_name="knowledge-base",
            object_name="documents/second.txt",
            chunk_index=0,
        )

        self.assertNotEqual(first, second)

    def test_separates_different_chunk_indexes(self) -> None:
        first = build_chunk_id(
            bucket_name="knowledge-base",
            object_name="documents/sample.txt",
            chunk_index=0,
        )
        second = build_chunk_id(
            bucket_name="knowledge-base",
            object_name="documents/sample.txt",
            chunk_index=1,
        )

        self.assertNotEqual(first, second)

    def test_rejects_invalid_identity_parts(self) -> None:
        invalid_arguments = [
            {
                "bucket_name": " ",
                "object_name": "documents/sample.txt",
                "chunk_index": 0,
            },
            {
                "bucket_name": "knowledge-base",
                "object_name": " ",
                "chunk_index": 0,
            },
            {
                "bucket_name": "knowledge-base",
                "object_name": "documents/sample.txt",
                "chunk_index": -1,
            },
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    build_chunk_id(**arguments)


class BuildKnowledgeChunksSchemaTests(unittest.TestCase):
    def test_builds_expected_schema(self) -> None:
        schema = build_knowledge_chunks_schema(embedding_dimension=1024)
        schema_dict = schema.to_dict()
        fields = schema_dict["fields"]

        self.assertEqual(
            [field["name"] for field in fields],
            [
                "chunk_id",
                "bucket_name",
                "object_name",
                "document_sha256",
                "chunk_index",
                "chunk_sha256",
                "text",
                "embedding",
            ],
        )
        self.assertEqual(
            [field["name"] for field in fields if field.get("is_primary")],
            ["chunk_id"],
        )
        self.assertFalse(schema_dict["enable_dynamic_field"])

        embedding = next(
            field for field in fields if field["name"] == "embedding"
        )
        self.assertEqual(embedding["params"]["dim"], 1024)

    def test_rejects_non_positive_embedding_dimension(self) -> None:
        for embedding_dimension in (0, -1):
            with self.subTest(embedding_dimension=embedding_dimension):
                with self.assertRaisesRegex(
                    ValueError,
                    "embedding_dimension must be > 0",
                ):
                    build_knowledge_chunks_schema(
                        embedding_dimension=embedding_dimension,
                    )


class EnsureKnowledgeChunksCollectionTests(unittest.TestCase):
    def test_creates_missing_collection(self) -> None:
        client = MagicMock(spec=MilvusClient)
        client.has_collection.return_value = False

        created = ensure_knowledge_chunks_collection(
            client,
            collection_name="knowledge_chunks",
            embedding_dimension=1024,
        )

        self.assertTrue(created)
        client.has_collection.assert_called_once_with(
            collection_name="knowledge_chunks",
        )
        client.prepare_index_params.return_value.add_index.assert_called_once_with(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        client.create_collection.assert_called_once()

    def test_keeps_existing_collection(self) -> None:
        client = MagicMock(spec=MilvusClient)
        client.has_collection.return_value = True

        created = ensure_knowledge_chunks_collection(
            client,
            collection_name="knowledge_chunks",
            embedding_dimension=1024,
        )

        self.assertFalse(created)
        client.prepare_index_params.assert_not_called()
        client.create_collection.assert_not_called()


class UpsertKnowledgeChunksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock(spec=MilvusClient)
        self.documents = [
            {
                "bucket_name": "knowledge-base",
                "object_name": "documents/sample.txt",
                "document_sha256": "a" * 64,
                "chunk_index": 0,
                "chunk_sha256": "b" * 64,
                "text": "第一段知识",
            },
            {
                "bucket_name": "knowledge-base",
                "object_name": "documents/sample.txt",
                "document_sha256": "a" * 64,
                "chunk_index": 1,
                "chunk_sha256": "c" * 64,
                "text": "第二段知识",
            },
        ]
        self.embeddings = [[0.1, 0.2], [0.3, 0.4]]

    def test_upserts_complete_entities_and_returns_chunk_ids(self) -> None:
        self.client.upsert.return_value = {"upsert_count": 2}

        chunk_ids = upsert_knowledge_chunks(
            self.client,
            collection_name="knowledge_chunks",
            chunk_documents=self.documents,
            embeddings=self.embeddings,
        )

        expected_ids = [
            build_chunk_id(
                bucket_name=document["bucket_name"],
                object_name=document["object_name"],
                chunk_index=document["chunk_index"],
            )
            for document in self.documents
        ]
        self.assertEqual(chunk_ids, expected_ids)
        self.client.upsert.assert_called_once_with(
            collection_name="knowledge_chunks",
            data=[
                {
                    "chunk_id": expected_ids[0],
                    **self.documents[0],
                    "embedding": [0.1, 0.2],
                },
                {
                    "chunk_id": expected_ids[1],
                    **self.documents[1],
                    "embedding": [0.3, 0.4],
                },
            ],
        )

    def test_returns_empty_list_without_calling_milvus(self) -> None:
        chunk_ids = upsert_knowledge_chunks(
            self.client,
            collection_name="knowledge_chunks",
            chunk_documents=[],
            embeddings=[],
        )

        self.assertEqual(chunk_ids, [])
        self.client.upsert.assert_not_called()

    def test_rejects_input_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            upsert_knowledge_chunks(
                self.client,
                collection_name="knowledge_chunks",
                chunk_documents=self.documents,
                embeddings=self.embeddings[:1],
            )

        self.client.upsert.assert_not_called()

    def test_rejects_unexpected_upsert_count(self) -> None:
        self.client.upsert.return_value = {"upsert_count": 1}

        with self.assertRaisesRegex(RuntimeError, "upsert数量"):
            upsert_knowledge_chunks(
                self.client,
                collection_name="knowledge_chunks",
                chunk_documents=self.documents,
                embeddings=self.embeddings,
            )


class DeleteStaleKnowledgeChunksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock(spec=MilvusClient)

    def test_deletes_chunks_from_first_stale_index(self) -> None:
        result = delete_stale_knowledge_chunks(
            self.client,
            collection_name="knowledge_chunks",
            bucket_name="knowledge-base",
            object_name="documents/store-operations.txt",
            first_stale_chunk_index=3,
        )

        self.assertIsNone(result)
        self.client.delete.assert_called_once_with(
            collection_name="knowledge_chunks",
            filter=(
                'bucket_name == "knowledge-base" and '
                'object_name == "documents/store-operations.txt" and '
                "chunk_index >= 3"
            ),
        )

    def test_rejects_invalid_inputs_without_calling_milvus(self) -> None:
        invalid_cases = [
            {"collection_name": ""},
            {"bucket_name": "  "},
            {"object_name": ""},
            {"first_stale_chunk_index": -1},
        ]

        valid_arguments = {
            "collection_name": "knowledge_chunks",
            "bucket_name": "knowledge-base",
            "object_name": "documents/store-operations.txt",
            "first_stale_chunk_index": 3,
        }

        for invalid_values in invalid_cases:
            with self.subTest(invalid_values=invalid_values):
                arguments = valid_arguments | invalid_values

                with self.assertRaises(ValueError):
                    delete_stale_knowledge_chunks(
                        self.client,
                        **arguments,
                    )

        self.client.delete.assert_not_called()


class SearchKnowledgeChunksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock(spec=MilvusClient)
        self.query_embedding = [0.1, 0.2]

    def test_searches_embedding_field_and_returns_single_result_group(self) -> None:
        hits = [
            {
                "id": "chunk-id",
                "distance": 0.95,
                "entity": {
                    "chunk_id": "chunk-id",
                    "bucket_name": "knowledge-base",
                    "object_name": "documents/sample.txt",
                    "chunk_index": 0,
                    "text": "第一段知识",
                },
            }
        ]
        self.client.search.return_value = [hits]

        results = search_knowledge_chunks(
            self.client,
            collection_name="knowledge_chunks",
            query_embedding=self.query_embedding,
            expected_dimension=2,
            limit=3,
        )

        self.assertEqual(results, hits)
        self.client.search.assert_called_once_with(
            collection_name="knowledge_chunks",
            data=[[0.1, 0.2]],
            anns_field="embedding",
            search_params={"metric_type": "COSINE"},
            limit=3,
            output_fields=[
                "chunk_id",
                "bucket_name",
                "object_name",
                "chunk_index",
                "text",
            ],
            consistency_level="Strong",
        )

    def test_rejects_invalid_search_inputs_without_calling_milvus(self) -> None:
        invalid_arguments = [
            {
                "collection_name": " ",
                "query_embedding": self.query_embedding,
                "expected_dimension": 2,
                "limit": 3,
            },
            {
                "collection_name": "knowledge_chunks",
                "query_embedding": self.query_embedding,
                "expected_dimension": 0,
                "limit": 3,
            },
            {
                "collection_name": "knowledge_chunks",
                "query_embedding": self.query_embedding,
                "expected_dimension": 3,
                "limit": 3,
            },
            {
                "collection_name": "knowledge_chunks",
                "query_embedding": self.query_embedding,
                "expected_dimension": 2,
                "limit": 0,
            },
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    search_knowledge_chunks(self.client, **arguments)

        self.client.search.assert_not_called()

    def test_rejects_unexpected_search_result_group_count(self) -> None:
        for search_result in ([], [[], []]):
            with self.subTest(search_result=search_result):
                self.client.search.return_value = search_result

                with self.assertRaisesRegex(RuntimeError, "结果组数量"):
                    search_knowledge_chunks(
                        self.client,
                        collection_name="knowledge_chunks",
                        query_embedding=self.query_embedding,
                        expected_dimension=2,
                        limit=3,
                    )

    def test_accepts_similarity_boundaries_and_rejects_out_of_range(self) -> None:
        self.client.search.return_value = [[]]

        for minimum_similarity in (-1.0, 1.0):
            with self.subTest(minimum_similarity=minimum_similarity):
                results = search_knowledge_chunks(
                    self.client,
                    collection_name="knowledge_chunks",
                    query_embedding=self.query_embedding,
                    expected_dimension=2,
                    minimum_similarity=minimum_similarity,
                )

                self.assertEqual(results, [])

        self.client.reset_mock()

        for minimum_similarity in (-1.01, 1.01):
            with self.subTest(minimum_similarity=minimum_similarity):
                with self.assertRaisesRegex(ValueError, "between -1 and 1"):
                    search_knowledge_chunks(
                        self.client,
                        collection_name="knowledge_chunks",
                        query_embedding=self.query_embedding,
                        expected_dimension=2,
                        minimum_similarity=minimum_similarity,
                    )

        self.client.search.assert_not_called()

    def test_rejects_result_without_distance(self) -> None:
        self.client.search.return_value = [
            [
                {
                    "id": "chunk-id",
                    "entity": {"text": "第一段知识"},
                }
            ]
        ]

        with self.assertRaisesRegex(RuntimeError, "缺少 distance"):
            search_knowledge_chunks(
                self.client,
                collection_name="knowledge_chunks",
                query_embedding=self.query_embedding,
                expected_dimension=2,
            )

    def test_filters_results_below_minimum_similarity_inclusively(self) -> None:
        accepted = {"id": "accepted", "distance": 0.5, "entity": {}}
        rejected = {"id": "rejected", "distance": 0.499, "entity": {}}
        self.client.search.return_value = [[accepted, rejected]]

        results = search_knowledge_chunks(
            self.client,
            collection_name="knowledge_chunks",
            query_embedding=self.query_embedding,
            expected_dimension=2,
            minimum_similarity=0.5,
        )

        self.assertEqual(results, [accepted])

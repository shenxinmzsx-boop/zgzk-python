import unittest
from argparse import Namespace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from knowledge.main import (
    answer_question,
    build_knowledge_base,
    embed_query_text,
    index_document_chunks,
    main,
    parse_cli_args,
    read_query_text,
    retrieve_knowledge_chunks,
    sync_document_to_mongodb,
    upload_and_verify_source,
)


class UploadAndVerifySourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_data = "知识库内容".encode("utf-8")
        self.source_sha256 = sha256(self.source_data).hexdigest()

    def test_uploads_and_returns_verified_download(self) -> None:
        settings = MagicMock()
        settings.minio_bucket_name = "knowledge-base"
        minio_client = MagicMock()

        with (
            patch(
                "knowledge.main.create_minio_client",
                return_value=minio_client,
            ) as create_client_mock,
            patch(
                "knowledge.main.ensure_minio_bucket",
                return_value=True,
            ) as ensure_bucket_mock,
            patch("knowledge.main.upload_bytes") as upload_mock,
            patch(
                "knowledge.main.download_bytes",
                return_value=self.source_data,
            ) as download_mock,
            patch("builtins.print"),
        ):
            downloaded_data = upload_and_verify_source(
                settings,
                object_name="documents/sample.txt",
                content_type="text/plain; charset=utf-8",
                source_data=self.source_data,
                source_sha256=self.source_sha256,
            )

        self.assertEqual(downloaded_data, self.source_data)
        create_client_mock.assert_called_once_with(settings)
        ensure_bucket_mock.assert_called_once_with(
            client=minio_client,
            bucket_name="knowledge-base",
        )
        upload_mock.assert_called_once_with(
            client=minio_client,
            bucket_name="knowledge-base",
            object_name="documents/sample.txt",
            data=self.source_data,
            content_type="text/plain; charset=utf-8",
        )
        download_mock.assert_called_once_with(
            client=minio_client,
            bucket_name="knowledge-base",
            object_name="documents/sample.txt",
        )

    def test_rejects_download_with_different_sha256(self) -> None:
        settings = MagicMock()
        settings.minio_bucket_name = "knowledge-base"

        with (
            patch("knowledge.main.create_minio_client"),
            patch("knowledge.main.ensure_minio_bucket", return_value=False),
            patch("knowledge.main.upload_bytes"),
            patch("knowledge.main.download_bytes", return_value=b"changed"),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 不一致"):
                upload_and_verify_source(
                    settings,
                    object_name="documents/sample.txt",
                    content_type="text/plain; charset=utf-8",
                    source_data=self.source_data,
                    source_sha256=self.source_sha256,
                )


class BuildKnowledgeBaseTests(unittest.TestCase):
    def test_rejects_blank_text_before_upload(self) -> None:
        settings = MagicMock()
        blank_contents = [
            b"",
            b"  \n\t",
            "　\n　".encode("utf-8"),
        ]

        with TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "blank.txt"

            for source_data in blank_contents:
                with self.subTest(source_data=source_data):
                    source_path.write_bytes(source_data)

                    with (
                        patch(
                            "knowledge.main.upload_and_verify_source"
                        ) as upload_mock,
                        patch("knowledge.main.split_text") as split_text_mock,
                        patch(
                            "knowledge.main.sync_document_to_mongodb"
                        ) as sync_mongodb_mock,
                        patch(
                            "knowledge.main.index_document_chunks"
                        ) as index_mock,
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            "源文件内容不能为空",
                        ):
                            build_knowledge_base(
                                settings,
                                source_path=source_path,
                                object_name="documents/blank.txt",
                                content_type="text/plain; charset=utf-8",
                                chunk_size=30,
                                chunk_overlap=5,
                            )

                    upload_mock.assert_not_called()
                    split_text_mock.assert_not_called()
                    sync_mongodb_mock.assert_not_called()
                    index_mock.assert_not_called()

    def test_rejects_invalid_utf8_before_upload(self) -> None:
        settings = MagicMock()

        with TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "invalid.txt"
            source_path.write_bytes(b"\xff\xfe\x00\x80")

            with (
                patch(
                    "knowledge.main.upload_and_verify_source"
                ) as upload_mock,
                patch("knowledge.main.split_text") as split_text_mock,
                patch(
                    "knowledge.main.sync_document_to_mongodb"
                ) as sync_mongodb_mock,
                patch("knowledge.main.index_document_chunks") as index_mock,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "源文件不是有效的 UTF-8 文本",
                ):
                    build_knowledge_base(
                        settings,
                        source_path=source_path,
                        object_name="documents/invalid.txt",
                        content_type="text/plain; charset=utf-8",
                        chunk_size=30,
                        chunk_overlap=5,
                    )

            upload_mock.assert_not_called()
            split_text_mock.assert_not_called()
            sync_mongodb_mock.assert_not_called()
            index_mock.assert_not_called()

    def test_rejects_invalid_source_paths_before_read_or_upload(self) -> None:
        settings = MagicMock()

        with TemporaryDirectory() as temporary_directory:
            directory_path = Path(temporary_directory)
            missing_path = directory_path / "missing.txt"
            unsupported_path = directory_path / "document.pdf"
            unsupported_path.write_bytes(b"content")

            invalid_cases = [
                (missing_path, FileNotFoundError, "源文件不存在"),
                (directory_path, ValueError, "不是普通文件"),
                (unsupported_path, ValueError, "只支持.txt文件"),
            ]

            for source_path, error_type, message in invalid_cases:
                with self.subTest(source_path=source_path):
                    with (
                        patch(
                            "knowledge.main.read_file_with_sha256"
                        ) as read_file_mock,
                        patch(
                            "knowledge.main.upload_and_verify_source"
                        ) as upload_mock,
                    ):
                        with self.assertRaisesRegex(error_type, message):
                            build_knowledge_base(
                                settings,
                                source_path=source_path,
                                object_name="documents/source.txt",
                                content_type="text/plain; charset=utf-8",
                                chunk_size=30,
                                chunk_overlap=5,
                            )

                    read_file_mock.assert_not_called()
                    upload_mock.assert_not_called()

    def test_runs_source_storage_chunking_and_indexing_pipeline(self) -> None:
        settings = MagicMock()
        source_path = Path("examples/sample.txt")
        source_data = "第一段第二段".encode("utf-8")
        source_sha256 = sha256(source_data).hexdigest()
        chunks = ["第一段", "第二段"]
        chunk_documents = [
            {"chunk_index": 0, "text": "第一段"},
            {"chunk_index": 1, "text": "第二段"},
        ]

        with (
            patch("knowledge.main.validate_source_path") as validate_mock,
            patch(
                "knowledge.main.read_file_with_sha256",
                return_value=(source_data, source_sha256),
            ) as read_file_mock,
            patch(
                "knowledge.main.upload_and_verify_source",
                return_value=source_data,
            ) as upload_mock,
            patch(
                "knowledge.main.split_text",
                return_value=chunks,
            ) as split_text_mock,
            patch(
                "knowledge.main.sync_document_to_mongodb",
                return_value=chunk_documents,
            ) as sync_mongodb_mock,
            patch("knowledge.main.index_document_chunks") as index_mock,
        ):
            result = build_knowledge_base(
                settings,
                source_path=source_path,
                object_name="documents/sample.txt",
                content_type="text/plain; charset=utf-8",
                chunk_size=30,
                chunk_overlap=5,
            )

        self.assertIsNone(result)
        validate_mock.assert_called_once_with(source_path)
        read_file_mock.assert_called_once_with(source_path)
        upload_mock.assert_called_once_with(
            settings,
            object_name="documents/sample.txt",
            content_type="text/plain; charset=utf-8",
            source_data=source_data,
            source_sha256=source_sha256,
        )
        split_text_mock.assert_called_once_with(
            source_data.decode("utf-8"),
            chunk_size=30,
            overlap=5,
        )
        sync_mongodb_mock.assert_called_once_with(
            settings,
            object_name="documents/sample.txt",
            original_filename="sample.txt",
            content_type="text/plain; charset=utf-8",
            source_size=len(source_data),
            source_sha256=source_sha256,
            chunks=chunks,
        )
        index_mock.assert_called_once_with(
            settings,
            chunk_documents=chunk_documents,
        )

    def test_stops_pipeline_when_source_upload_fails(self) -> None:
        settings = MagicMock()
        source_path = Path("examples/sample.txt")
        source_data = b"content"

        with (
            patch("knowledge.main.validate_source_path"),
            patch(
                "knowledge.main.read_file_with_sha256",
                return_value=(source_data, "source-sha256"),
            ),
            patch(
                "knowledge.main.upload_and_verify_source",
                side_effect=RuntimeError("upload failed"),
            ),
            patch("knowledge.main.split_text") as split_text_mock,
            patch(
                "knowledge.main.sync_document_to_mongodb"
            ) as sync_mongodb_mock,
            patch("knowledge.main.index_document_chunks") as index_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "upload failed"):
                build_knowledge_base(
                    settings,
                    source_path=source_path,
                    object_name="documents/sample.txt",
                    content_type="text/plain; charset=utf-8",
                    chunk_size=30,
                    chunk_overlap=5,
                )

        split_text_mock.assert_not_called()
        sync_mongodb_mock.assert_not_called()
        index_mock.assert_not_called()


class AnswerQuestionTests(unittest.TestCase):
    def test_returns_no_answer_without_calling_chat_when_search_is_empty(
        self,
    ) -> None:
        settings = MagicMock()

        with (
            patch(
                "knowledge.main.retrieve_knowledge_chunks",
                return_value=[],
            ) as retrieve_mock,
            patch("knowledge.main.create_ai_client") as create_client_mock,
            patch("knowledge.main.generate_answer") as generate_mock,
        ):
            result = answer_question(
                settings,
                query_text="不存在的问题",
            )

        self.assertEqual(result, (None, []))
        retrieve_mock.assert_called_once_with(
            settings,
            query_text="不存在的问题",
        )
        create_client_mock.assert_not_called()
        generate_mock.assert_not_called()

    def test_builds_grounded_prompt_and_returns_answer_with_sources(self) -> None:
        settings = MagicMock()
        settings.chat_model_name = "chat-model"
        search_results = [
            {
                "entity": {
                    "text": "资料一",
                    "object_name": "documents/first.txt",
                    "chunk_index": 0,
                }
            },
            {
                "entity": {
                    "text": "资料二",
                    "object_name": "documents/second.txt",
                    "chunk_index": 3,
                }
            },
        ]
        ai_client_context = MagicMock()
        chat_client = ai_client_context.__enter__.return_value

        with (
            patch(
                "knowledge.main.retrieve_knowledge_chunks",
                return_value=search_results,
            ),
            patch(
                "knowledge.main.create_ai_client",
                return_value=ai_client_context,
            ),
            patch(
                "knowledge.main.generate_answer",
                return_value="答案 [来源：documents/first.txt，Chunk 0]",
            ) as generate_mock,
        ):
            answer, returned_results = answer_question(
                settings,
                query_text="用户问题",
            )

        self.assertEqual(
            answer,
            "答案 [来源：documents/first.txt，Chunk 0]",
        )
        self.assertIs(returned_results, search_results)
        generate_mock.assert_called_once()
        call_kwargs = generate_mock.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "chat-model")
        self.assertIs(generate_mock.call_args.args[0], chat_client)
        self.assertIn("只能根据提供的知识库资料回答", call_kwargs["messages"][0]["content"])
        self.assertIn(
            "必须原样引用知识库资料开头的来源标签",
            call_kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "[来源：documents/first.txt，Chunk 0]\n资料一",
            call_kwargs["messages"][1]["content"],
        )
        self.assertIn(
            "[来源：documents/second.txt，Chunk 3]\n资料二",
            call_kwargs["messages"][1]["content"],
        )
        self.assertIn("<question>\n用户问题\n</question>", call_kwargs["messages"][1]["content"])
        ai_client_context.__exit__.assert_called_once_with(None, None, None)


class CLICommandTests(unittest.TestCase):
    def test_parses_build_command_with_source_path_and_document_id(self) -> None:
        with patch(
            "sys.argv",
            [
                "knowledge",
                "build",
                "examples/custom_knowledge.txt",
                "--document-id",
                "store-operations",
            ],
        ):
            args = parse_cli_args()

        self.assertEqual(args.command, "build")
        self.assertEqual(
            args.source_path,
            Path("examples/custom_knowledge.txt"),
        )
        self.assertEqual(args.document_id, "store-operations")

    def test_requires_document_id_for_build_command(self) -> None:
        with (
            patch(
                "sys.argv",
                ["knowledge", "build", "examples/custom_knowledge.txt"],
            ),
            patch("sys.stderr"),
        ):
            with self.assertRaises(SystemExit) as raised:
                parse_cli_args()

        self.assertEqual(raised.exception.code, 2)

    def test_rejects_invalid_document_id(self) -> None:
        invalid_document_ids = [
            "Store-Operations",
            "store.operations",
            "a" * 65,
        ]

        for document_id in invalid_document_ids:
            with self.subTest(document_id=document_id):
                with (
                    patch(
                        "sys.argv",
                        [
                            "knowledge",
                            "build",
                            "examples/custom_knowledge.txt",
                            "--document-id",
                            document_id,
                        ],
                    ),
                    patch("sys.stderr"),
                ):
                    with self.assertRaises(SystemExit) as raised:
                        parse_cli_args()

                self.assertEqual(raised.exception.code, 2)

    def test_parses_ask_command_without_source_path(self) -> None:
        with patch("sys.argv", ["knowledge", "ask"]):
            args = parse_cli_args()

        self.assertEqual(args.command, "ask")
        self.assertFalse(hasattr(args, "source_path"))
        self.assertFalse(hasattr(args, "document_id"))

    def test_build_command_only_builds_knowledge_base(self) -> None:
        settings = MagicMock()

        with (
            patch(
                "knowledge.main.parse_cli_args",
                return_value=Namespace(
                    command="build",
                    source_path=Path("examples/custom_knowledge.txt"),
                    document_id="store-operations",
                ),
            ),
            patch("knowledge.main.get_settings", return_value=settings),
            patch("knowledge.main.build_knowledge_base") as build_mock,
            patch("knowledge.main.read_query_text") as read_query_mock,
            patch("knowledge.main.answer_question") as answer_mock,
        ):
            main()

        build_mock.assert_called_once_with(
            settings,
            source_path=Path("examples/custom_knowledge.txt").resolve(),
            object_name="documents/store-operations.txt",
            content_type="text/plain; charset=utf-8",
            chunk_size=30,
            chunk_overlap=5,
        )
        read_query_mock.assert_not_called()
        answer_mock.assert_not_called()

    def test_ask_command_only_answers_question(self) -> None:
        settings = MagicMock()

        with (
            patch(
                "knowledge.main.parse_cli_args",
                return_value=Namespace(command="ask"),
            ),
            patch("knowledge.main.get_settings", return_value=settings),
            patch("knowledge.main.build_knowledge_base") as build_mock,
            patch(
                "knowledge.main.read_query_text",
                return_value="原始文件存储在哪里？",
            ),
            patch(
                "knowledge.main.answer_question",
                return_value=(None, []),
            ) as answer_mock,
            patch("builtins.print"),
        ):
            main()

        build_mock.assert_not_called()
        answer_mock.assert_called_once_with(
            settings=settings,
            query_text="原始文件存储在哪里？",
        )


class ReadQueryTextTests(unittest.TestCase):
    @patch("builtins.input", return_value="  原始文件存储在哪里？  ")
    def test_returns_trimmed_query_text(self, input_mock) -> None:
        query_text = read_query_text()

        self.assertEqual(query_text, "原始文件存储在哪里？")
        input_mock.assert_called_once_with("\n请输入问题: ")

    def test_rejects_empty_query_text(self) -> None:
        for user_input in ("", "   "):
            with self.subTest(user_input=user_input):
                with patch("builtins.input", return_value=user_input):
                    with self.assertRaisesRegex(ValueError, "问题不能为空"):
                        read_query_text()


class EmbedQueryTextTests(unittest.TestCase):
    def test_embeds_one_query_and_returns_its_vector(self) -> None:
        settings = MagicMock()
        settings.embedding_model_name = "embedding-model"
        settings.milvus_embedding_dimension = 3
        ai_client_context = MagicMock()
        embedding_client = ai_client_context.__enter__.return_value

        with (
            patch(
                "knowledge.main.create_ai_client",
                return_value=ai_client_context,
            ) as create_client_mock,
            patch(
                "knowledge.main.embed_texts",
                return_value=[[0.1, 0.2, 0.3]],
            ) as embed_texts_mock,
        ):
            result = embed_query_text(
                settings,
                query_text="原始文件存储在哪里？",
            )

        self.assertEqual(result, [0.1, 0.2, 0.3])
        create_client_mock.assert_called_once_with(settings)
        embed_texts_mock.assert_called_once_with(
            embedding_client,
            model="embedding-model",
            texts=["原始文件存储在哪里？"],
            expected_dimension=3,
        )
        ai_client_context.__exit__.assert_called_once_with(None, None, None)


class IndexDocumentChunksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = MagicMock()
        self.settings.embedding_model_name = "embedding-model"
        self.settings.milvus_embedding_dimension = 3
        self.settings.milvus_collection_name = "knowledge_chunks"
        self.chunk_documents = [
            {
                "text": "chunk-0",
                "bucket_name": "knowledge-base",
                "object_name": "documents/sample.txt",
                "document_sha256": "document-sha256",
                "chunk_index": 0,
                "chunk_sha256": "chunk-sha256-0",
            }
        ]
        self.ai_client_context = MagicMock()
        self.embedding_client = self.ai_client_context.__enter__.return_value
        self.milvus_client = MagicMock()

    def test_embeds_upserts_verifies_and_closes_clients(self) -> None:
        self.milvus_client.query.return_value = [{"chunk_id": "chunk-id"}]

        with (
            patch(
                "knowledge.main.create_ai_client",
                return_value=self.ai_client_context,
            ),
            patch(
                "knowledge.main.embed_texts",
                return_value=[[0.1, 0.2, 0.3]],
            ) as embed_texts_mock,
            patch(
                "knowledge.main.create_milvus_client",
                return_value=self.milvus_client,
            ),
            patch(
                "knowledge.main.ensure_knowledge_chunks_collection",
                return_value=True,
            ),
            patch(
                "knowledge.main.upsert_knowledge_chunks",
                return_value=["chunk-id"],
            ) as upsert_mock,
            patch(
                "knowledge.main.delete_stale_knowledge_chunks",
            ) as delete_stale_mock,
            patch("builtins.print"),
        ):
            result = index_document_chunks(
                self.settings,
                chunk_documents=self.chunk_documents,
            )

        self.assertIsNone(result)
        embed_texts_mock.assert_called_once_with(
            self.embedding_client,
            model="embedding-model",
            texts=["chunk-0"],
            expected_dimension=3,
        )
        upsert_mock.assert_called_once_with(
            self.milvus_client,
            collection_name="knowledge_chunks",
            chunk_documents=self.chunk_documents,
            embeddings=[[0.1, 0.2, 0.3]],
        )
        delete_stale_mock.assert_called_once_with(
            self.milvus_client,
            collection_name="knowledge_chunks",
            bucket_name="knowledge-base",
            object_name="documents/sample.txt",
            first_stale_chunk_index=1,
        )
        self.ai_client_context.__exit__.assert_called_once_with(None, None, None)
        self.milvus_client.close.assert_called_once_with()

    def test_closes_milvus_when_stale_chunk_cleanup_fails(self) -> None:
        with (
            patch(
                "knowledge.main.create_ai_client",
                return_value=self.ai_client_context,
            ),
            patch(
                "knowledge.main.embed_texts",
                return_value=[[0.1, 0.2, 0.3]],
            ),
            patch(
                "knowledge.main.create_milvus_client",
                return_value=self.milvus_client,
            ),
            patch(
                "knowledge.main.ensure_knowledge_chunks_collection",
                return_value=False,
            ),
            patch(
                "knowledge.main.upsert_knowledge_chunks",
                return_value=["chunk-id"],
            ),
            patch(
                "knowledge.main.delete_stale_knowledge_chunks",
                side_effect=RuntimeError("delete failed"),
            ),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(RuntimeError, "delete failed"):
                index_document_chunks(
                    self.settings,
                    chunk_documents=self.chunk_documents,
                )

        self.milvus_client.query.assert_not_called()
        self.milvus_client.close.assert_called_once_with()

    def test_closes_milvus_when_readback_count_mismatches(self) -> None:
        self.milvus_client.query.return_value = []

        with (
            patch(
                "knowledge.main.create_ai_client",
                return_value=self.ai_client_context,
            ),
            patch(
                "knowledge.main.embed_texts",
                return_value=[[0.1, 0.2, 0.3]],
            ),
            patch(
                "knowledge.main.create_milvus_client",
                return_value=self.milvus_client,
            ),
            patch(
                "knowledge.main.ensure_knowledge_chunks_collection",
                return_value=False,
            ),
            patch(
                "knowledge.main.upsert_knowledge_chunks",
                return_value=["chunk-id"],
            ),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Entity 数量"):
                index_document_chunks(
                    self.settings,
                    chunk_documents=self.chunk_documents,
                )

        self.milvus_client.close.assert_called_once_with()


class RetrieveKnowledgeChunksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = MagicMock()
        self.settings.milvus_collection_name = "knowledge_chunks"
        self.settings.milvus_embedding_dimension = 3
        self.milvus_client = MagicMock()

    def test_embeds_query_searches_and_closes_client(self) -> None:
        search_results = [
            {
                "distance": 0.9,
                "entity": {
                    "chunk_index": 0,
                    "text": "知识库内容",
                },
            }
        ]

        with (
            patch(
                "knowledge.main.embed_query_text",
                return_value=[0.1, 0.2, 0.3],
            ) as embed_query_mock,
            patch(
                "knowledge.main.create_milvus_client",
                return_value=self.milvus_client,
            ),
            patch(
                "knowledge.main.search_knowledge_chunks",
                return_value=search_results,
            ) as search_mock,
        ):
            result = retrieve_knowledge_chunks(
                self.settings,
                query_text="原始文件存储在哪里？",
            )

        self.assertEqual(result, search_results)
        embed_query_mock.assert_called_once_with(
            self.settings,
            query_text="原始文件存储在哪里？",
        )
        search_mock.assert_called_once_with(
            self.milvus_client,
            collection_name="knowledge_chunks",
            query_embedding=[0.1, 0.2, 0.3],
            expected_dimension=3,
            limit=3,
            minimum_similarity=0.5,
        )
        self.milvus_client.close.assert_called_once_with()

    def test_closes_client_when_search_fails(self) -> None:
        with (
            patch(
                "knowledge.main.embed_query_text",
                return_value=[0.1, 0.2, 0.3],
            ),
            patch(
                "knowledge.main.create_milvus_client",
                return_value=self.milvus_client,
            ),
            patch(
                "knowledge.main.search_knowledge_chunks",
                side_effect=RuntimeError("search failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "search failed"):
                retrieve_knowledge_chunks(
                    self.settings,
                    query_text="原始文件存储在哪里？",
                )

        self.milvus_client.close.assert_called_once_with()


class SyncDocumentToMongoDBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = MagicMock()
        self.settings.minio_bucket_name = "knowledge-base"
        self.settings.mongo_database = "knowledge"
        self.settings.mongo_documents_collection = "documents"
        self.settings.mongo_document_chunks_collection = "document_chunks"

        self.mongo_client = MagicMock()
        self.database = MagicMock()
        self.metadata_collection = MagicMock()
        self.chunk_collection = MagicMock()
        self.mongo_client.__getitem__.return_value = self.database
        self.database.__getitem__.side_effect = {
            "documents": self.metadata_collection,
            "document_chunks": self.chunk_collection,
        }.__getitem__

        self.chunk_documents = [
            {
                "text": f"chunk-{index}",
                "bucket_name": "knowledge-base",
                "object_name": "documents/sample.txt",
                "document_sha256": "document-sha256",
                "chunk_index": index,
                "chunk_sha256": f"chunk-sha256-{index}",
            }
            for index in range(5)
        ]

    def test_syncs_and_returns_embedding_chunk_documents(self) -> None:
        cursor = MagicMock()
        cursor.sort.return_value = cursor
        cursor.__iter__.return_value = iter(self.chunk_documents)
        self.chunk_collection.count_documents.return_value = 5
        self.chunk_collection.find.return_value = cursor

        with (
            patch(
                "knowledge.main.create_mongo_client",
                return_value=self.mongo_client,
            ),
            patch("knowledge.main.ping_mongodb") as ping_mock,
            patch("knowledge.main.ensure_document_metadata_indexes"),
            patch("knowledge.main.ensure_document_chunk_indexes"),
            patch("knowledge.main.upsert_document_metadata"),
            patch(
                "knowledge.main.find_document_metadata",
                return_value={"object_name": "documents/sample.txt"},
            ),
            patch("knowledge.main.sync_document_chunks") as sync_chunks_mock,
        ):
            result = sync_document_to_mongodb(
                self.settings,
                object_name="documents/sample.txt",
                original_filename="sample.txt",
                content_type="text/plain; charset=utf-8",
                source_size=128,
                source_sha256="document-sha256",
                chunks=[
                    "chunk-0",
                    "chunk-1",
                    "chunk-2",
                    "chunk-3",
                    "chunk-4",
                ],
            )

        self.assertEqual(result, self.chunk_documents)
        ping_mock.assert_called_once_with(self.mongo_client)
        sync_chunks_mock.assert_called_once_with(
            self.chunk_collection,
            bucket_name="knowledge-base",
            object_name="documents/sample.txt",
            document_sha256="document-sha256",
            chunks=[
                "chunk-0",
                "chunk-1",
                "chunk-2",
                "chunk-3",
                "chunk-4",
            ],
        )
        cursor.limit.assert_not_called()
        self.mongo_client.close.assert_called_once_with()

    def test_does_not_sync_chunks_when_metadata_readback_fails(self) -> None:
        with (
            patch(
                "knowledge.main.create_mongo_client",
                return_value=self.mongo_client,
            ),
            patch("knowledge.main.ping_mongodb"),
            patch("knowledge.main.ensure_document_metadata_indexes"),
            patch("knowledge.main.ensure_document_chunk_indexes"),
            patch("knowledge.main.upsert_document_metadata"),
            patch(
                "knowledge.main.find_document_metadata",
                return_value=None,
            ),
            patch("knowledge.main.sync_document_chunks") as sync_chunks_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "未查询到刚写入的对象元数据"):
                sync_document_to_mongodb(
                    self.settings,
                    object_name="documents/sample.txt",
                    original_filename="sample.txt",
                    content_type="text/plain; charset=utf-8",
                    source_size=128,
                    source_sha256="document-sha256",
                    chunks=["chunk-0", "chunk-1", "chunk-2"],
                )

        sync_chunks_mock.assert_not_called()
        self.mongo_client.close.assert_called_once_with()

    def test_rejects_stored_chunk_count_mismatch_and_closes_client(self) -> None:
        self.chunk_collection.count_documents.return_value = 2

        with (
            patch(
                "knowledge.main.create_mongo_client",
                return_value=self.mongo_client,
            ),
            patch("knowledge.main.ping_mongodb"),
            patch("knowledge.main.ensure_document_metadata_indexes"),
            patch("knowledge.main.ensure_document_chunk_indexes"),
            patch("knowledge.main.upsert_document_metadata"),
            patch(
                "knowledge.main.find_document_metadata",
                return_value={"object_name": "documents/sample.txt"},
            ),
            patch("knowledge.main.sync_document_chunks"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Chunk数量"):
                sync_document_to_mongodb(
                    self.settings,
                    object_name="documents/sample.txt",
                    original_filename="sample.txt",
                    content_type="text/plain; charset=utf-8",
                    source_size=128,
                    source_sha256="document-sha256",
                    chunks=["chunk-0", "chunk-1", "chunk-2"],
                )

        self.mongo_client.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

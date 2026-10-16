import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from knowledge.utils.embeddings import embed_texts


class EmbedTextsTests(unittest.TestCase):
    def test_returns_empty_list_without_calling_api(self) -> None:
        client = MagicMock()

        embeddings = embed_texts(
            client,
            model="BAAI/bge-m3",
            texts=[],
            expected_dimension=3,
        )

        self.assertEqual(embeddings, [])
        client.embeddings.create.assert_not_called()

    def test_rejects_blank_text_without_calling_api(self) -> None:
        client = MagicMock()

        with self.assertRaisesRegex(ValueError, "待向量化文本不能为空"):
            embed_texts(
                client,
                model="BAAI/bge-m3",
                texts=["有效文本", "   "],
                expected_dimension=3,
            )

        client.embeddings.create.assert_not_called()

    def test_calls_api_and_restores_input_order(self) -> None:
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[4.0, 5.0, 6.0]),
                SimpleNamespace(index=0, embedding=[1.0, 2.0, 3.0]),
            ],
        )

        embeddings = embed_texts(
            client,
            model="BAAI/bge-m3",
            texts=["第一段", "第二段"],
            expected_dimension=3,
        )

        self.assertEqual(
            embeddings,
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )
        client.embeddings.create.assert_called_once_with(
            model="BAAI/bge-m3",
            input=["第一段", "第二段"],
            encoding_format="float",
        )

    def test_rejects_response_count_mismatch(self) -> None:
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 2.0, 3.0])],
        )

        with self.assertRaisesRegex(RuntimeError, "向量数量和文本数量不一致"):
            embed_texts(
                client,
                model="BAAI/bge-m3",
                texts=["第一段", "第二段"],
                expected_dimension=3,
            )

    def test_rejects_response_dimension_mismatch(self) -> None:
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 2.0])],
        )

        with self.assertRaisesRegex(RuntimeError, "向量维度不是 3"):
            embed_texts(
                client,
                model="BAAI/bge-m3",
                texts=["第一段"],
                expected_dimension=3,
            )


if __name__ == "__main__":
    unittest.main()

import unittest

from knowledge.text_chunking import split_text


class SplitTextTests(unittest.TestCase):
    def test_split_text_with_overlap(self) -> None:
        result = split_text(
            "ABCDEFGHIJ",
            chunk_size=4,
            overlap=1,
        )
        self.assertEqual(result, ["ABCD", "DEFG", "GHIJ"])

    def test_returns_empty_list_for_empty_text(self) -> None:
        result = split_text("", chunk_size=4, overlap=1)
        self.assertEqual(result, [])

    def test_rejects_non_positive_chunk_size(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "chunk_size must be > 0",
        ):
            split_text("ABC", chunk_size=0, overlap=0)

    def test_rejects_negative_overlap(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "overlap must be >= 0",
        ):
            split_text("ABC", chunk_size=4, overlap=-1)

    def test_rejects_overlap_equal_to_chunk_size(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "overlap must be < chunk_size",
        ):
            split_text("ABC", chunk_size=4, overlap=4)

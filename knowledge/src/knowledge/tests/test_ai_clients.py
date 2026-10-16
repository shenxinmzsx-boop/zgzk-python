import unittest
from unittest.mock import MagicMock, patch

from knowledge.utils.client.ai_clients import create_ai_client


class CreateAiClientTests(unittest.TestCase):
    @patch("knowledge.utils.client.ai_clients.OpenAI")
    def test_creates_client_from_settings(self, openai_class: MagicMock) -> None:
        settings = MagicMock()
        settings.embedding_api_key.get_secret_value.return_value = "test-key"
        settings.embedding_api_base_url = "https://example.invalid/v1"
        settings.embedding_timeout_seconds = 30

        client = create_ai_client(settings)

        self.assertIs(client, openai_class.return_value)
        settings.embedding_api_key.get_secret_value.assert_called_once_with()
        openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            timeout=30,
        )


if __name__ == "__main__":
    unittest.main()

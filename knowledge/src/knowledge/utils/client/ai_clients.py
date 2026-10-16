from openai import OpenAI

from knowledge.core.settings import Settings


def create_ai_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.embedding_api_key.get_secret_value(),
        base_url=settings.embedding_api_base_url,
        timeout=settings.embedding_timeout_seconds,
    )

from collections.abc import Sequence

from openai import OpenAI


def embed_texts(
    client: OpenAI,
    *,
    model: str,
    texts: Sequence[str],
    expected_dimension: int,
) -> list[list[float]]:
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("待向量化文本不能为空")

    response = client.embeddings.create(
        model=model,
        input=list(texts),
        encoding_format="float"
    )

    ordered_items = sorted(response.data, key=lambda item:item.index)
    embeddings = [item.embedding for item in ordered_items]

    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding API 返回的向量数量和文本数量不一致")

    if any(len(embedding) != expected_dimension for embedding in embeddings):
        raise RuntimeError(
            f"Embedding API 返回的向量维度不是 {expected_dimension}"
        )

    return embeddings

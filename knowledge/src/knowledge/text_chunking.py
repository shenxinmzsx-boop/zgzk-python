def split_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    if not text:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start += step

    return chunks

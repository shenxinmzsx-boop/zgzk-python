from openai import OpenAI

def generate_answer(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    if not model.strip():
        raise ValueError("model must not be empty")

    if not messages:
        raise ValueError("messages must not be empty")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    if not response.choices:
        raise RuntimeError("Chat API 未返回回答选项")

    if len(response.choices) != 1:
        raise RuntimeError(
            f"Chat API 应返回1个回答，实际返回{len(response.choices)}"
        )

    answer = response.choices[0].message.content

    if answer is None or not answer.strip():
        raise RuntimeError("Chat API 返回的回答为空")

    return answer.strip()
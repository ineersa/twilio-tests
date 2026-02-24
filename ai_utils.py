from collections.abc import Generator, Sequence

from openai import OpenAI


def stream_ai_response(
    client: OpenAI, model: str, messages: Sequence[dict[str, str]]
) -> Generator[str]:
    stream = client.chat.completions.create(
        model=model,
        messages=list(messages),
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        if content:
            yield content

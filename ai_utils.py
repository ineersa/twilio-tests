import json
from collections.abc import Generator, Sequence
from typing import Any

from openai import OpenAI

ALLOWED_DATA_CENTER_COMPANIES = [
    "DPR Construction",
    "AECOM",
    "Turner Construction",
    "Fortis Construction",
    "Skanska USA",
]


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


def validate_questionnaire_answer(
    client: OpenAI,
    model: str,
    question: dict[str, str],
    answer: str,
) -> tuple[bool, Any, str]:
    allowed_companies_text = ", ".join(ALLOWED_DATA_CENTER_COMPANIES)
    system_prompt = (
        "You validate spoken questionnaire answers. "
        "Return JSON only with keys: is_valid, normalized_answer, error_message."
    )
    user_prompt = (
        "Question metadata:\n"
        f"- id: {question['id']}\n"
        f"- type: {question['type']}\n"
        f"- prompt: {question['prompt']}\n\n"
        f"User answer: {answer}\n\n"
        "Validation rules:\n"
        "1) type=name: valid only if a person name is present. normalized_answer must be the name string.\n"
        "2) type=yes_no: valid only for yes/no intent. normalized_answer must be exactly \"yes\" or \"no\".\n"
        "3) type=scale_1_10: valid only if answer clearly maps to integer 1..10. normalized_answer must be integer.\n"
        "4) type=topic_text: valid only if answer is related to the asked topic. normalized_answer should be cleaned text.\n"
        "5) type=topic_entity (for id=data_center_construction_company): extract the company name from answer. "
        "normalized_answer must be an object: {\"company_name\": string, \"is_in_list\": boolean}. "
        f"Set is_in_list=true only if company_name matches one of: {allowed_companies_text}.\n"
        "6) If no company can be identified, mark invalid.\n"
        "7) If invalid, set normalized_answer to null and provide a short spoken-friendly error_message.\n"
        "8) If valid, set error_message to empty string."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content or "{}"
    parsed = json.loads(content)
    is_valid = bool(parsed.get("is_valid", False))
    normalized_answer = parsed.get("normalized_answer")
    error_message = str(parsed.get("error_message", "")).strip()
    if not is_valid and not error_message:
        error_message = "Please try answering that question again."
    return is_valid, normalized_answer, error_message

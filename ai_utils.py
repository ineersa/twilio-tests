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


def classify_transcription_compliance(
    client: OpenAI,
    model: str,
    transcript_text: str,
    recent_transcripts: Sequence[str] | None = None,
) -> tuple[bool, list[str]]:
    cleaned_transcript = transcript_text.strip()
    if not cleaned_transcript:
        return False, []

    cleaned_context = [
        snippet.strip()
        for snippet in (recent_transcripts or [])
        if isinstance(snippet, str) and snippet.strip()
    ]
    context_block = "\n".join(
        f"{index}. {snippet}" for index, snippet in enumerate(cleaned_context, start=1)
    )

    system_prompt = (
        "You are a strict compliance classifier for call transcript snippets. "
        "Detect whether the transcript contains prohibited topics.\n"
        "The topic examples below are illustrative and non-exhaustive. "
        "You must generalize to similar terms (for example, acne treatment should be treated as a medical/treatment concept).\n\n"
        "Prohibited topics:\n"
        "1) Medical & Symptom Concepts:\n"
        "- Any symptoms (e.g., headache, back pain, inflammation)\n"
        "- Any conditions/diseases (e.g., arthritis, chronic pain, migraine, acne)\n"
        "- Any treatment/use-case discussion (e.g., acute pain, long-term use, treatment plans)\n"
        "2) Scientific & Development Concepts:\n"
        "- Mechanism of action or how a product works\n"
        "- Dose ceiling, dosing limits, dosage guidance\n"
        "- Efficacy/effectiveness claims\n"
        "- Clinical claims, study/trial results, comparative claims\n"
        "3) Financial / Planning Concepts:\n"
        "- Margin, profitability\n"
        "- Costs, budget, spend\n"
        "- Pricing, discounts, monetization\n"
        "- Forecasts, projections, revenue planning\n"
        "4) Proper nouns:\n"
        "- Company names\n"
        "- Product names\n"
        "- Competitor names\n\n"
        "Return JSON only with keys: is_compliance_violation, compliance_violations.\n"
        "Rules:\n"
        "- Classify the current transcript snippet using prior snippets only as supporting context.\n"
        "- If prior context has violations but current snippet is unrelated, return false.\n"
        "- Set is_compliance_violation=true only when the current snippet conveys a prohibited topic.\n"
        "- compliance_violations must be an array of exact words/phrases copied from the transcript.\n"
        "- If there is no violation, return false and an empty array.\n"
        "- Do not include any keys besides is_compliance_violation and compliance_violations."
    )
    user_prompt = (
        f"Current transcript snippet:\n{cleaned_transcript}\n\n"
        "Recent prior transcript snippets (oldest to newest):\n"
        f"{context_block or 'none'}"
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
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return False, []

    is_compliance_violation = bool(parsed.get("is_compliance_violation", False))
    raw_violations = parsed.get("compliance_violations")
    if not isinstance(raw_violations, list):
        return False, []

    validation_source = "\n".join([cleaned_transcript, *cleaned_context]).lower()
    seen: set[str] = set()
    compliance_violations: list[str] = []
    for item in raw_violations:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if not candidate:
            continue
        # Keep only phrases that appear in provided transcript context to reduce hallucinations.
        if candidate.lower() not in validation_source:
            continue
        dedupe_key = candidate.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        compliance_violations.append(candidate)

    if not compliance_violations:
        return False, []

    return is_compliance_violation or bool(compliance_violations), compliance_violations


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

import json
import logging
import os
from asyncio import Task, create_task, sleep
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from fastapi import WebSocket
from twilio.rest import Client

QUESTIONNAIRE: list[dict[str, str]] = [
    {"id": "name", "prompt": "What is your name?", "type": "name"},
    {
        "id": "public_company_employment",
        "prompt": "Are you currently employed at a public company? Please answer yes or no.",
        "type": "yes_no",
    },
    {
        "id": "data_center_construction_experience",
        "prompt": "What is your experience in data center construction?",
        "type": "topic_text",
    },
    {
        "id": "data_center_construction_company",
        "prompt": "What datacenter construction company you are working in?",
        "type": "topic_entity",
    },
    {
        "id": "location_selection_experience",
        "prompt": (
            "On a scale of one to ten, what is your experience dealing with "
            "data center location selection?"
        ),
        "type": "scale_1_10",
    },
]
COMPLETION_MESSAGE = (
    "Thank you. The questionnaire is complete. You can hang up safely now."
)
ANSWERS_DIR = Path(__file__).resolve().parent / "answers"
SILENCE_TIMEOUT_SECONDS = 30
_TWILIO_CLIENT: Client | None = None


async def send_assistant_message(websocket: WebSocket, text: str) -> None:
    await websocket.send_text(
        json.dumps(
            {
                "type": "text",
                "token": text,
                "last": False,
            }
        )
    )
    await websocket.send_text(
        json.dumps(
            {
                "type": "text",
                "token": "",
                "last": True,
            }
        )
    )


def _persist_answers(
    call_sid: str,
    answers: dict[str, Any],
    status: str,
    termination_reason: str,
    logger: logging.Logger,
) -> None:
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANSWERS_DIR / f"{call_sid}.json"
    payload = {
        "call_sid": call_sid,
        "answers": answers,
        "status": status,
        "termination_reason": termination_reason,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved questionnaire answers: %s", output_path)


def _get_twilio_client(logger: logging.Logger) -> Client | None:
    global _TWILIO_CLIENT  # noqa: PLW0603
    if _TWILIO_CLIENT is not None:
        return _TWILIO_CLIENT

    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    if not account_sid or not auth_token:
        logger.error("Missing TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN; cannot hang up call")
        return None

    _TWILIO_CLIENT = Client(account_sid, auth_token)
    return _TWILIO_CLIENT


def _hangup_call_via_twilio(call_sid: str, speech_text: str, logger: logging.Logger) -> None:
    twilio_client = _get_twilio_client(logger)
    if twilio_client is None:
        return

    safe_message = escape(speech_text)
    twiml = f"<Response><Say>{safe_message}</Say><Hangup/></Response>"
    twilio_client.calls(call_sid).update(twiml=twiml)
    logger.info("Twilio hangup sent for call: %s", call_sid)


def _cancel_silence_timer(state: dict[str, Any]) -> None:
    silence_task = state.get("silence_task")
    if isinstance(silence_task, Task) and not silence_task.done():
        silence_task.cancel()
    state["silence_task"] = None


async def _terminate_call(
    call_sid: str,
    call_states: dict[str, dict[str, Any]],
    logger: logging.Logger,
    *,
    spoken_message: str,
    status: str,
    termination_reason: str,
) -> None:
    state = call_states.get(call_sid)
    if not state:
        logger.info("Termination skipped for unknown call: %s", call_sid)
        return
    if state.get("terminated"):
        logger.info("Termination already processed for call: %s", call_sid)
        return

    state["terminated"] = True
    state["termination_reason"] = termination_reason
    state["status"] = status
    state["completed"] = status == "completed"
    _cancel_silence_timer(state)

    answers = state.setdefault("answers", {})
    _persist_answers(call_sid, answers, status, termination_reason, logger)

    try:
        _hangup_call_via_twilio(call_sid, spoken_message, logger)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send Twilio hangup for call: %s", call_sid)


async def _silence_watchdog(
    call_sid: str, call_states: dict[str, dict[str, Any]], logger: logging.Logger
) -> None:
    try:
        await sleep(SILENCE_TIMEOUT_SECONDS)
    except Exception:  # noqa: BLE001
        return

    state = call_states.get(call_sid)
    if not state or state.get("terminated"):
        return

    logger.info("Silence timeout reached for call: %s", call_sid)
    await _terminate_call(
        call_sid,
        call_states,
        logger,
        spoken_message="Ahoy!",
        status="terminated",
        termination_reason="silence",
    )


def _start_or_reset_silence_timer(
    call_sid: str, call_states: dict[str, dict[str, Any]], logger: logging.Logger
) -> None:
    state = call_states.get(call_sid)
    if not state or state.get("terminated"):
        return
    _cancel_silence_timer(state)
    state["silence_task"] = create_task(_silence_watchdog(call_sid, call_states, logger))


async def send_current_question(
    call_sid: str,
    call_states: dict[str, dict[str, Any]],
    websocket: WebSocket,
    logger: logging.Logger,
) -> None:
    state = call_states.get(call_sid)
    if not state:
        logger.warning("No active state found while sending question for call: %s", call_sid)
        return
    if state.get("terminated"):
        return

    question_index = int(state.get("question_index", 0))
    if question_index >= len(QUESTIONNAIRE):
        await _terminate_call(
            call_sid,
            call_states,
            logger,
            spoken_message=COMPLETION_MESSAGE,
            status="completed",
            termination_reason="completed",
        )
        return

    prompt = QUESTIONNAIRE[question_index]["prompt"]
    await send_assistant_message(websocket, prompt)
    _start_or_reset_silence_timer(call_sid, call_states, logger)
    logger.info("Asked question %s for call: %s", question_index + 1, call_sid)


def parse_ws_message(raw: str, logger: logging.Logger) -> dict[str, Any] | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Received invalid JSON on websocket")
        return None


def handle_setup_message(
    message: dict[str, Any],
    call_states: dict[str, dict[str, Any]],
    logger: logging.Logger,
) -> str | None:
    call_sid = str(message.get("callSid", "")).strip()
    if not call_sid:
        logger.warning("setup message missing callSid")
        return None

    call_states[call_sid] = {
        "question_index": 0,
        "answers": {},
        "completed": False,
        "status": "in_progress",
        "termination_reason": "",
        "invalid_attempts_current_question": 0,
        "terminated": False,
        "silence_task": None,
    }
    logger.info("Setup for call: %s", call_sid)
    return call_sid


async def handle_prompt_message(
    message: dict[str, Any],
    call_sid: str | None,
    call_states: dict[str, dict[str, Any]],
    answer_validator: Callable[[dict[str, str], str], tuple[bool, Any, str]],
    websocket: WebSocket,
    logger: logging.Logger,
) -> None:
    if not call_sid:
        logger.warning("prompt received before setup")
        return

    voice_prompt = str(message.get("voicePrompt", "")).strip()
    if not voice_prompt:
        logger.warning("prompt message missing voicePrompt")
        return

    state = call_states.get(call_sid)
    if state is None:
        logger.warning("prompt received for unknown call: %s", call_sid)
        return

    if state.get("terminated"):
        logger.info("Ignoring prompt for terminated call: %s", call_sid)
        return

    _cancel_silence_timer(state)

    question_index = int(state.get("question_index", 0))
    if question_index >= len(QUESTIONNAIRE):
        await _terminate_call(
            call_sid,
            call_states,
            logger,
            spoken_message=COMPLETION_MESSAGE,
            status="completed",
            termination_reason="completed",
        )
        return

    question = QUESTIONNAIRE[question_index]
    try:
        is_valid, normalized_answer, error_message = answer_validator(question, voice_prompt)
    except Exception:  # noqa: BLE001
        logger.exception("LLM answer validation failed for call: %s", call_sid)
        await send_assistant_message(
            websocket,
            f"I had trouble validating that response. {question['prompt']}",
        )
        return

    if not is_valid:
        invalid_attempts = int(state.get("invalid_attempts_current_question", 0)) + 1
        state["invalid_attempts_current_question"] = invalid_attempts
        if invalid_attempts >= 3:
            logger.info("Reached invalid answer limit for call: %s", call_sid)
            await _terminate_call(
                call_sid,
                call_states,
                logger,
                spoken_message="invalid answers provided",
                status="terminated",
                termination_reason="invalid_answers",
            )
            return

        feedback = error_message or "That answer does not match this question."
        await send_assistant_message(
            websocket,
            f"{feedback} {question['prompt']}",
        )
        _start_or_reset_silence_timer(call_sid, call_states, logger)
        logger.info("Invalid answer for call %s at question %s", call_sid, question_index + 1)
        return

    state["invalid_attempts_current_question"] = 0
    answers = state.setdefault("answers", {})
    answers[question["id"]] = normalized_answer
    next_index = question_index + 1
    state["question_index"] = next_index

    if next_index >= len(QUESTIONNAIRE):
        await _terminate_call(
            call_sid,
            call_states,
            logger,
            spoken_message=COMPLETION_MESSAGE,
            status="completed",
            termination_reason="completed",
        )
        return

    next_prompt = QUESTIONNAIRE[next_index]["prompt"]
    await send_assistant_message(websocket, next_prompt)
    _start_or_reset_silence_timer(call_sid, call_states, logger)
    logger.info("Asked question %s for call: %s", next_index + 1, call_sid)


def handle_interrupt_message(
    message: dict[str, Any],
    call_sid: str | None,
    call_states: dict[str, dict[str, Any]],
    logger: logging.Logger,
) -> None:
    if not call_sid:
        logger.warning("interrupt received before setup")
        return

    utterance_until_interrupt = str(message.get("utteranceUntilInterrupt", "")).strip()
    if not utterance_until_interrupt:
        logger.info("Interrupt received for call %s without utterance payload", call_sid)
        return

    state = call_states.get(call_sid)
    if not state:
        logger.info("Interrupt received for unknown call: %s", call_sid)
        return

    logger.info(
        "Interrupt noted for call %s while on question %s",
        call_sid,
        int(state.get("question_index", 0)) + 1,
    )


def cleanup_session(
    call_states: dict[str, dict[str, Any]], call_sid: str | None
) -> None:
    if call_sid:
        state = call_states.pop(call_sid, None)
        if state:
            _cancel_silence_timer(state)

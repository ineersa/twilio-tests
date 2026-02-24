import json
import logging
from collections.abc import Callable, Iterable
from typing import Any

from fastapi import WebSocket


def parse_ws_message(raw: str, logger: logging.Logger) -> dict[str, Any] | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Received invalid JSON on websocket")
        return None


def handle_setup_message(
    message: dict[str, Any],
    sessions: dict[str, list[dict[str, str]]],
    system_prompt: str,
    logger: logging.Logger,
) -> str | None:
    call_sid = str(message.get("callSid", "")).strip()
    if not call_sid:
        logger.warning("setup message missing callSid")
        return None

    sessions[call_sid] = [{"role": "system", "content": system_prompt}]
    logger.info("Setup for call: %s", call_sid)
    return call_sid


async def handle_prompt_message(
    message: dict[str, Any],
    call_sid: str | None,
    sessions: dict[str, list[dict[str, str]]],
    system_prompt: str,
    ai_response: Callable[[list[dict[str, str]]], Iterable[str]],
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

    conversation = sessions.setdefault(
        call_sid,
        [{"role": "system", "content": system_prompt}],
    )
    conversation.append({"role": "user", "content": voice_prompt})

    assistant_segments: list[str] = []
    try:
        for token in ai_response(conversation):
            assistant_segments.append(token)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "text",
                        "token": token,
                        "last": False,
                    }
                )
            )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to stream OpenAI response")
        fallback = "I ran into a temporary issue while generating a response."
        assistant_segments = [fallback]
        await websocket.send_text(
            json.dumps(
                {
                    "type": "text",
                    "token": fallback,
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
    response_text = "".join(assistant_segments).strip()
    if not response_text:
        response_text = "I am sorry, I could not generate a response."
    conversation.append({"role": "assistant", "content": response_text})
    logger.info("Sent response for call: %s", call_sid)


def handle_interrupt_message(
    message: dict[str, Any],
    call_sid: str | None,
    sessions: dict[str, list[dict[str, str]]],
    logger: logging.Logger,
) -> None:
    if not call_sid:
        logger.warning("interrupt received before setup")
        return

    utterance_until_interrupt = str(message.get("utteranceUntilInterrupt", "")).strip()
    if not utterance_until_interrupt:
        logger.info("Interrupt received for call %s without utterance payload", call_sid)
        return

    conversation = sessions.get(call_sid)
    if not conversation:
        logger.info("Interrupt received for unknown call: %s", call_sid)
        return

    interrupted_index = -1
    for index in range(len(conversation) - 1, -1, -1):
        entry = conversation[index]
        if (
            entry.get("role") == "assistant"
            and utterance_until_interrupt in entry.get("content", "")
        ):
            interrupted_index = index
            break

    if interrupted_index == -1:
        logger.info("No matching assistant turn found for interrupt on call: %s", call_sid)
        return

    interrupted_message = conversation[interrupted_index]
    original_content = interrupted_message.get("content", "")
    interrupt_position = original_content.find(utterance_until_interrupt)
    truncated_content = original_content[
        : interrupt_position + len(utterance_until_interrupt)
    ]
    conversation[interrupted_index] = {
        "role": "assistant",
        "content": truncated_content,
    }
    sessions[call_sid] = [
        item
        for i, item in enumerate(conversation)
        if not (i > interrupted_index and item.get("role") == "assistant")
    ]
    logger.info("Interrupt processed for call: %s", call_sid)


def cleanup_session(
    sessions: dict[str, list[dict[str, str]]], call_sid: str | None
) -> None:
    if call_sid:
        sessions.pop(call_sid, None)

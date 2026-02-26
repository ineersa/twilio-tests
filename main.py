import json
import logging
import os
from collections import deque
from functools import partial
from time import monotonic
from typing import Any
from urllib.parse import parse_qsl

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from openai import OpenAI

from ai_utils import validate_questionnaire_answer
from relay_handlers import (
    cleanup_session,
    handle_interrupt_message,
    handle_prompt_message,
    handle_setup_message,
    parse_ws_message,
    send_current_question,
)

load_dotenv(".env")
load_dotenv(".env.local", override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", "8080"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
WELCOME_GREETING = (
    "Hi! I am a voice assistant powered by Twilio and Open A I ."
)

app = FastAPI(title="Twilio ConversationRelay with FastAPI")

# In-memory questionnaire state keyed by Twilio callSid.
call_states: dict[str, dict[str, object]] = {}
compliance_clients: set[WebSocket] = set()
RECENT_TRANSCRIPT_TTL_SECONDS = 3.0
_recent_transcript_keys: deque[tuple[float, str]] = deque()
_recent_transcript_key_set: set[str] = set()

def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required. Put shared examples in .env and secrets in .env.local."
        )
    return value


DOMAIN = require_env("NGROK_URL")
OPENAI_API_KEY = require_env("OPENAI_API_KEY")
WS_URL = f"wss://{DOMAIN}/ws"
client = OpenAI(api_key=OPENAI_API_KEY)
answer_validator = partial(validate_questionnaire_answer, client, OPENAI_MODEL)


async def _parse_transcription_payload(request: Request) -> dict[str, Any]:
    raw_body = await request.body()
    decoded_body = raw_body.decode("utf-8", errors="replace")
    content_type = request.headers.get("content-type", "").lower()

    if "application/x-www-form-urlencoded" in content_type:
        pairs = parse_qsl(decoded_body, keep_blank_values=True)
        return {key: value for key, value in pairs}

    if "application/json" in content_type:
        try:
            json_payload = json.loads(decoded_body)
        except json.JSONDecodeError:
            return {"raw_body": decoded_body}
        if isinstance(json_payload, dict):
            return json_payload
        return {"payload": json_payload}

    pairs = parse_qsl(decoded_body, keep_blank_values=True)
    if pairs:
        return {key: value for key, value in pairs}

    try:
        json_payload = json.loads(decoded_body)
    except json.JSONDecodeError:
        return {"raw_body": decoded_body}
    if isinstance(json_payload, dict):
        return json_payload
    return {"payload": json_payload}


async def _broadcast_to_compliance(payload: dict[str, Any]) -> int:
    if not compliance_clients:
        return 0

    message = json.dumps(payload)
    delivered_clients = 0
    disconnected_clients: list[WebSocket] = []

    for websocket in list(compliance_clients):
        try:
            await websocket.send_text(message)
            delivered_clients += 1
        except Exception:  # noqa: BLE001
            disconnected_clients.append(websocket)

    for websocket in disconnected_clients:
        compliance_clients.discard(websocket)

    return delivered_clients


def _extract_transcript_text(payload: dict[str, Any]) -> str:
    transcription_data = payload.get("TranscriptionData")

    if isinstance(transcription_data, dict):
        transcript = transcription_data.get("transcript", "")
        return str(transcript)

    if isinstance(transcription_data, str):
        try:
            parsed_data = json.loads(transcription_data)
        except json.JSONDecodeError:
            return transcription_data

        if isinstance(parsed_data, dict):
            transcript = parsed_data.get("transcript", "")
            return str(transcript)
        return str(parsed_data)

    return ""


def _is_duplicate_transcription_content(payload: dict[str, Any]) -> bool:
    if payload.get("TranscriptionEvent") != "transcription-content":
        return False

    if str(payload.get("Final", "")).lower() == "false":
        return False

    transcript = _extract_transcript_text(payload)
    normalized_transcript = " ".join(transcript.strip().lower().split())
    if not normalized_transcript:
        return False

    now = monotonic()
    while _recent_transcript_keys and now - _recent_transcript_keys[0][0] > RECENT_TRANSCRIPT_TTL_SECONDS:
        _, expired_key = _recent_transcript_keys.popleft()
        _recent_transcript_key_set.discard(expired_key)

    if normalized_transcript in _recent_transcript_key_set:
        return True

    _recent_transcript_keys.append((now, normalized_transcript))
    _recent_transcript_key_set.add(normalized_transcript)
    return False


@app.get("/twiml")
async def twiml() -> Response:
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<ConversationRelay url="{WS_URL}" welcomeGreeting="{WELCOME_GREETING}" interruptSensitivity="high" />'
        "</Connect>"
        "</Response>"
    )
    return Response(content=body, media_type="text/xml")


@app.post("/transcription")
async def transcription_webhook(request: Request) -> dict[str, object]:
    payload = await _parse_transcription_payload(request)
    if _is_duplicate_transcription_content(payload):
        logger.info("Transcription webhook duplicate skipped payload=%s", payload)
        return {"ok": True, "delivered_clients": 0}

    delivered_clients = await _broadcast_to_compliance(payload)

    logger.info("Transcription webhook payload=%s", payload)

    return {"ok": True, "delivered_clients": delivered_clients}


@app.websocket("/compliance")
async def compliance_ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    compliance_clients.add(websocket)
    logger.info("Compliance websocket connected. Active clients: %s", len(compliance_clients))

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        logger.info("Compliance websocket disconnected")
    finally:
        compliance_clients.discard(websocket)
        logger.info("Compliance websocket active clients: %s", len(compliance_clients))


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    call_sid: str | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            message = parse_ws_message(raw, logger)
            if message is None:
                continue

            logger.info("Received websocket message: %s", message)
            message_type = message.get("type")

            if message_type == "setup":
                new_call_sid = handle_setup_message(message, call_states, logger)
                if new_call_sid:
                    call_sid = new_call_sid
                    await send_current_question(call_sid, call_states, websocket, logger)

            elif message_type == "prompt":
                await handle_prompt_message(
                    message=message,
                    call_sid=call_sid,
                    call_states=call_states,
                    answer_validator=answer_validator,
                    websocket=websocket,
                    logger=logger,
                )

            elif message_type == "interrupt":
                handle_interrupt_message(message, call_sid, call_states, logger)

            else:
                logger.warning("Unknown message type: %s", message_type)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for call: %s", call_sid)
    finally:
        cleanup_session(call_states, call_sid)


def main() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)


if __name__ == "__main__":
    main()

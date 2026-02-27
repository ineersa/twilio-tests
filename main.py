import json
import logging
import os
from collections import deque
from functools import partial
from typing import Any
from urllib.parse import parse_qsl

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from openai import OpenAI

from ai_utils import classify_transcription_compliance, validate_questionnaire_answer
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
inbound_transcript_context: dict[str, deque[str]] = {}

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
compliance_classifier = partial(classify_transcription_compliance, client, OPENAI_MODEL)


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
    if isinstance(transcription_data, str):
        try:
            transcription_data = json.loads(transcription_data)
        except json.JSONDecodeError:
            return ""
    if not isinstance(transcription_data, dict):
        return ""
    transcript = transcription_data.get("transcript")
    if not isinstance(transcript, str):
        return ""
    return transcript.strip()


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
    if payload.get("Track") != "inbound_track":
        logger.info("Transcription webhook skipped non-inbound payload=%s", payload)
        return {"ok": True, "delivered_clients": 0}

    call_sid = str(payload.get("CallSid", "")).strip()
    context_window: list[str] = []
    if call_sid:
        call_context = inbound_transcript_context.get(call_sid)
        if call_context is None:
            call_context = deque(maxlen=3)
            inbound_transcript_context[call_sid] = call_context
        context_window = list(call_context)

    enriched_payload = dict(payload)
    is_compliance_violation = False
    compliance_violations: list[str] = []
    transcript_text = _extract_transcript_text(payload)
    if transcript_text:
        try:
            is_compliance_violation, compliance_violations = compliance_classifier(
                transcript_text,
                context_window,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Compliance classification failed for payload=%s", payload)
    else:
        logger.warning("Transcription payload missing transcript text: %s", payload)

    enriched_payload["IsComplianceViolation"] = is_compliance_violation
    enriched_payload["ComplianceViolations"] = compliance_violations

    delivered_clients = await _broadcast_to_compliance(enriched_payload)

    if transcript_text and call_sid:
        inbound_transcript_context[call_sid].append(transcript_text)

    logger.info("Transcription webhook payload=%s", enriched_payload)

    return {"ok": True, "delivered_clients": delivered_clients}


@app.post("/summary")
async def summary_webhook(request: Request) -> dict[str, object]:
    payload = await _parse_transcription_payload(request)
    delivered_clients = await _broadcast_to_compliance(payload)

    call_sid = str(payload.get("CallSid", "")).strip()
    if call_sid:
        inbound_transcript_context.pop(call_sid, None)

    logger.info("Summary webhook payload=%s", payload)

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

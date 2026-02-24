import logging
import os
from functools import partial

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from openai import OpenAI

from ai_utils import stream_ai_response
from relay_handlers import (
    cleanup_session,
    handle_interrupt_message,
    handle_prompt_message,
    handle_setup_message,
    parse_ws_message,
)

load_dotenv(".env")
load_dotenv(".env.local", override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", "8080"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
WELCOME_GREETING = (
    "Hi! I am a voice assistant powered by Twilio and Open A I . Ask me anything!"
)
SYSTEM_PROMPT = (
    "You are a helpful assistant. This conversation is being translated to voice, "
    "so answer carefully. When you respond, please spell out all numbers, for "
    "example twenty not 20. Do not include emojis in your responses. Do not include "
    "bullet points, asterisks, or special symbols."
)

app = FastAPI(title="Twilio ConversationRelay with FastAPI")

# In-memory session store keyed by Twilio callSid.
sessions: dict[str, list[dict[str, str]]] = {}

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
ai_response = partial(stream_ai_response, client, OPENAI_MODEL)


@app.get("/twiml")
async def twiml() -> Response:
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<ConversationRelay url="{WS_URL}" welcomeGreeting="{WELCOME_GREETING}" />'
        "</Connect>"
        "</Response>"
    )
    return Response(content=body, media_type="text/xml")


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
                new_call_sid = handle_setup_message(message, sessions, SYSTEM_PROMPT, logger)
                if new_call_sid:
                    call_sid = new_call_sid

            elif message_type == "prompt":
                await handle_prompt_message(
                    message=message,
                    call_sid=call_sid,
                    sessions=sessions,
                    system_prompt=SYSTEM_PROMPT,
                    ai_response=ai_response,
                    websocket=websocket,
                    logger=logger,
                )

            elif message_type == "interrupt":
                handle_interrupt_message(message, call_sid, sessions, logger)

            else:
                logger.warning("Unknown message type: %s", message_type)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for call: %s", call_sid)
    finally:
        cleanup_session(sessions, call_sid)


def main() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)


if __name__ == "__main__":
    main()

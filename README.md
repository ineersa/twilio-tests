# Twilio ConversationRelay + FastAPI

Python implementation of a Twilio Voice + ConversationRelay bridge that routes caller speech to OpenAI and returns synthesized responses through Twilio.

This project follows the flow from Twilio's ConversationRelay tutorial, adapted to FastAPI and Python:
- https://www.twilio.com/en-us/blog/developers/tutorials/product/integrate-openai-twilio-voice-using-conversationrelay

## What is implemented

- `GET /twiml`: returns TwiML with `<ConversationRelay ... />`
- `WebSocket /ws`: handles Twilio ConversationRelay events (`setup`, `prompt`, `interrupt`)
- In-memory per-call conversation state keyed by `callSid`
- OpenAI token streaming for each prompt (`last:false` chunks + final `last:true`)
- Explicit environment loading order:
  - `.env`
  - `.env.local` with override enabled

## Project structure

- `main.py`
  - FastAPI app and routes
  - required env validation
  - OpenAI client init
  - app startup entrypoint (`main()`)
- `relay_handlers.py`
  - WebSocket message parsing
  - message-specific handlers
  - session cleanup helper
- `.env`
  - tracked example values (non-secret template)
- `.env.local`
  - local secrets only (ignored by git)

## Runtime behavior (technical details)

### 1) Twilio webhook step

Twilio calls:
- `GET https://<your-domain>/twiml`

Server returns TwiML:
- `<Connect><ConversationRelay url="wss://<NGROK_URL>/ws" .../></Connect>`

Nuance:
- `NGROK_URL` must be the host only (no `https://`), because server builds `wss://{NGROK_URL}/ws`.

### 2) WebSocket event loop

In `ws_endpoint`:
- accepts socket
- receives text payload
- parses JSON using `parse_ws_message(...)`
- logs inbound message
- dispatches by `message["type"]`

Current handled event types:
- `setup`
  - reads `callSid`
  - initializes session with system prompt
- `prompt`
  - reads `voicePrompt`
  - appends caller message to session
  - streams OpenAI tokens to Twilio incrementally
  - appends final joined assistant response to session
  - sends Twilio response payloads:
    - `{"type":"text","token":"...","last":false}` for each streamed token
    - `{"type":"text","token":"","last":true}` when stream completes
- `interrupt`
  - uses `utteranceUntilInterrupt` to truncate assistant history to what caller heard

On disconnect:
- session is deleted from memory.

### 3) Session model

`sessions` is an in-memory dictionary:
- key: `callSid`
- value: list of OpenAI-style messages

Format example:
- `{"role": "system", "content": "..."}`
- `{"role": "user", "content": "..."}`
- `{"role": "assistant", "content": "..."}`

Nuances:
- state is process-local and ephemeral
- restarting server clears all active conversation context
- multiple workers would not share this state

### 4) Environment loading and precedence

At startup:
- `load_dotenv(".env")`
- `load_dotenv(".env.local", override=True)`

Therefore:
- `.env.local` values override `.env`
- intended workflow:
  - `.env` = shared, non-secret examples
  - `.env.local` = machine-specific secrets

Required variables are validated at import/startup:
- `NGROK_URL`
- `OPENAI_API_KEY`

If missing, app fails fast with a clear `RuntimeError`.

Optional variables:
- `PORT` (default `8080`)
- `OPENAI_MODEL` (default `gpt-4o-mini`)

## Setup

## 1) Configure env files

Edit `.env` (tracked template):
- `PORT=8080`
- `NGROK_URL=...`
- `OPENAI_MODEL=gpt-4o-mini`
- placeholder for `OPENAI_API_KEY`

Create `.env.local` (not tracked) with real secrets:

```bash
OPENAI_API_KEY=sk-...
```

## 2) Install dependencies

```bash
uv sync
```

## 3) Run app

```bash
uv run python main.py
```

Alternative:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8080
```

## 4) Expose local app (ngrok)

```bash
ngrok http 8080
```

Take the public host (for example `1234abcd.ngrok.app`) and set it as:
- `NGROK_URL=1234abcd.ngrok.app`

## 5) Configure Twilio number

In Twilio Console for your phone number:
- "A call comes in" -> Webhook
- URL: `https://<ngrok-host>/twiml`
- Method: `GET`

## Known limitations / follow-up opportunities

- streaming runs inside the prompt handler, so interruption affects history for subsequent turns
- in-memory sessions only (no Redis/DB)
- no auth/allowlist on WebSocket endpoint yet

## Troubleshooting

- `RuntimeError` for missing env var:
  - verify `.env`/`.env.local` and variable names exactly match expected keys
- Twilio does not connect to WebSocket:
  - confirm `NGROK_URL` has no scheme
  - confirm Twilio webhook URL points to `/twiml` with `GET`
- assistant replies are generic error text:
  - check OpenAI key and model in env

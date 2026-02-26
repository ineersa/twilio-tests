#!/usr/bin/env python3
import argparse
import asyncio
import json
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse

import websockets


def _build_ws_url(target: str, path: str, use_tls: bool) -> str:
    normalized_target = target.strip()

    if normalized_target.startswith("http://"):
        normalized_target = "ws://" + normalized_target[len("http://") :]
    elif normalized_target.startswith("https://"):
        normalized_target = "wss://" + normalized_target[len("https://") :]

    if normalized_target.startswith("ws://") or normalized_target.startswith("wss://"):
        parsed = urlparse(normalized_target)
        final_path = parsed.path if parsed.path and parsed.path != "/" else path
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                final_path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

    scheme = "wss" if use_tls else "ws"
    host = normalized_target.strip("/")
    return f"{scheme}://{host}{path}"


def _format_message(message: str, raw: bool) -> str:
    if raw:
        return message

    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return message

    return json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True)


async def _listen(url: str, raw: bool) -> None:
    print(f"Connecting to {url}")
    async with websockets.connect(url) as websocket:
        print("Connected. Waiting for compliance events... (Ctrl+C to stop)")
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                print(f"[{datetime.now(timezone.utc).isoformat()}] <binary {len(message)} bytes>")
                continue

            timestamp = datetime.now(timezone.utc).isoformat()
            print(f"[{timestamp}]")
            print(_format_message(message, raw))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Listen to /compliance websocket events from your ngrok host."
    )
    parser.add_argument(
        "target",
        help="Ngrok host or full websocket URL (e.g. abc123.ngrok.app or wss://abc123.ngrok.app/compliance)",
    )
    parser.add_argument(
        "--path",
        default="/compliance",
        help="WebSocket path when target is only a host (default: /compliance)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Use ws:// instead of wss:// when target is only a host",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw payload without JSON formatting",
    )
    args = parser.parse_args()

    ws_url = _build_ws_url(args.target, args.path, use_tls=not args.insecure)
    try:
        asyncio.run(_listen(ws_url, raw=args.raw))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

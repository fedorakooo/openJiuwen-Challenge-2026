"""A local OpenAI-compatible endpoint for tests: records requests, answers from a script."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Responder = Callable[[dict[str, Any]], tuple[int, dict[str, Any]]]


def completion(
    content: str,
    prompt_tokens: int = 100,
    completion_tokens: int = 10,
    reasoning_content: str | None = None,
) -> tuple[int, dict[str, Any]]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return 200, {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class FakeProxy:
    """Handle to the running server."""

    def __init__(self, server: ThreadingHTTPServer) -> None:
        self._server = server
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.headers: list[dict[str, str]] = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"


@contextmanager
def fake_proxy(responder: Responder) -> Iterator[FakeProxy]:
    """Serve `/v1/chat/completions` on a free local port for the duration of the block."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - name fixed by http.server
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            proxy.requests.append(body)
            proxy.paths.append(self.path)
            proxy.headers.append({name.lower(): value for name, value in self.headers.items()})
            status, payload = responder(body)
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    proxy = FakeProxy(server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield proxy
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

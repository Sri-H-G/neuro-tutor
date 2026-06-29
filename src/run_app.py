"""Unified app: brain replay + WebSocket state + HTTP frontend + tutor API."""

import argparse
import asyncio
import json
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import websockets

from src.brain_loop import run_replay_loop
from src.config import ROOT_DIR
from src.decoder import StateDecoder
from src.shared_state import BrainState
from src.tutor import AdaptiveTutor

FRONTEND_DIR = ROOT_DIR / "frontend"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Shared app singletons (set before server starts)
_brain_state: BrainState | None = None
_tutor: AdaptiveTutor | None = None
_lesson_started: bool = False


class AppHandler(SimpleHTTPRequestHandler):
    """Serve frontend static files and tutor chat API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def log_message(self, format, *args):
        if args and args[0].startswith("GET /"):
            return  # quiet static requests
        super().log_message(format, *args)

    def do_GET(self):
        path = urlparse(self.path).path

        if path.startswith("/outputs/"):
            file_path = OUTPUTS_DIR / path.removeprefix("/outputs/")
            if file_path.is_file():
                self._serve_file(file_path)
                return
            self.send_error(404)
            return

        if path == "/api/status":
            self._json_response(self._status_payload())
            return

        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/chat":
            self._handle_chat()
            return

        if path == "/api/start":
            self._handle_start()
            return

        self.send_error(404)

    def _serve_file(self, file_path: Path):
        content_type = "application/octet-stream"
        if file_path.suffix == ".png":
            content_type = "image/png"
        elif file_path.suffix == ".json":
            content_type = "application/json"

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_response(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _status_payload(self) -> dict:
        global _lesson_started
        snap = _brain_state.snapshot()
        return {
            "engagement": snap.engagement,
            "load": snap.load,
            "mode": snap.mode,
            "calibrating": snap.calibrating,
            "timestamp": snap.timestamp,
            "lesson_started": _lesson_started,
            "has_api_key": bool(__import__("os").getenv("GEMINI_API_KEY")),
        }

    def _handle_chat(self):
        global _tutor
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            message = body.get("message", "").strip()
            if not message:
                self._json_response({"error": "message required"}, 400)
                return

            reply = _tutor.respond(message)
            snap = _brain_state.snapshot()
            self._json_response({
                "reply": reply,
                "mode": snap.mode,
                "engagement": snap.engagement,
                "load": snap.load,
            })
        except ValueError as exc:
            self._json_response({"error": str(exc)}, 500)
        except Exception as exc:
            self._json_response({"error": str(exc)}, 500)

    def _handle_start(self):
        global _lesson_started, _tutor
        try:
            reply = _tutor.start_lesson()
            _lesson_started = True
            snap = _brain_state.snapshot()
            self._json_response({
                "reply": reply,
                "mode": snap.mode,
                "engagement": snap.engagement,
                "load": snap.load,
            })
        except ValueError as exc:
            self._json_response({"error": str(exc)}, 500)
        except Exception as exc:
            self._json_response({"error": str(exc)}, 500)


async def websocket_handler(websocket):
    """Broadcast brain state to connected clients."""
    try:
        while True:
            snap = _brain_state.snapshot()
            payload = {
                "engagement": snap.engagement,
                "load": snap.load,
                "mode": snap.mode,
                "calibrating": snap.calibrating,
                "timestamp": snap.timestamp,
            }
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(0.25)
    except websockets.ConnectionClosed:
        pass


async def run_websocket(host: str, port: int):
    async with websockets.serve(websocket_handler, host, port):
        print(f"WebSocket  ws://{host}:{port}")
        await asyncio.Future()


def run_http(host: str, port: int):
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Frontend   http://{host}:{port}")
    server.serve_forever()


def main():
    global _brain_state, _tutor

    parser = argparse.ArgumentParser(description="Neuro GenAI Loop — full web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--speed", type=float, default=20.0,
                        help="EEG replay speed multiplier")
    parser.add_argument("--narrate", action="store_true",
                        help="Tutor explains adaptations")
    args = parser.parse_args()

    _brain_state = BrainState()
    _tutor = AdaptiveTutor(_brain_state, narrate=args.narrate)
    decoder = StateDecoder()
    stop_event = threading.Event()

    brain_thread = threading.Thread(
        target=run_replay_loop,
        args=(_brain_state, decoder, None, None, args.speed, stop_event),
        daemon=True,
    )
    brain_thread.start()
    print(f"Brain loop replaying rest→task at {args.speed}× speed")

    http_thread = threading.Thread(
        target=run_http,
        args=(args.host, args.http_port),
        daemon=True,
    )
    http_thread.start()

    print("\nOpen http://127.0.0.1:8080 in your browser.\n")

    try:
        asyncio.run(run_websocket(args.host, args.ws_port))
    except KeyboardInterrupt:
        stop_event.set()
        print("\nStopped.")


if __name__ == "__main__":
    main()

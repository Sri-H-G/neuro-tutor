"""WebSocket bridge: broadcast brain state to the frontend."""

import asyncio
import json

import websockets

from src.shared_state import BrainState


async def broadcast_state(
    brain_state: BrainState,
    host: str = "localhost",
    port: int = 8765,
) -> None:
    """Serve brain snapshots to connected clients ~4×/sec."""

    async def handler(websocket):
        try:
            while True:
                snap = brain_state.snapshot()
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

    async with websockets.serve(handler, host, port):
        print(f"WebSocket server on ws://{host}:{port}")
        await asyncio.Future()  # run forever

"""Thread-safe bridge between the brain loop and the tutor."""

import time
from dataclasses import dataclass
from threading import Lock

from src.decoder import DecodeResult


@dataclass
class BrainSnapshot:
    """Latest brain-state reading for the tutor or websocket clients."""

    engagement: float
    load: float
    mode: str
    calibrating: bool
    timestamp: float


class BrainState:
    """Shared state updated by the brain loop, read by the tutor."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._engagement = 0.0
        self._load = 0.0
        self._mode = "calibrating"
        self._calibrating = True
        self._timestamp = time.time()

    def update(self, result: DecodeResult, timestamp: float | None = None) -> None:
        with self._lock:
            self._engagement = result.engagement
            self._load = result.load
            self._mode = result.mode
            self._calibrating = result.calibrating
            self._timestamp = timestamp if timestamp is not None else time.time()

    def snapshot(self) -> BrainSnapshot:
        with self._lock:
            return BrainSnapshot(
                engagement=self._engagement,
                load=self._load,
                mode=self._mode,
                calibrating=self._calibrating,
                timestamp=self._timestamp,
            )

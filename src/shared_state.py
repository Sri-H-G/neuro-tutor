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
    alpha: float = 0.0
    theta: float = 0.0
    beta: float = 0.0


class BrainState:
    """Shared state updated by the brain loop, read by the tutor."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._engagement = 0.0
        self._load = 0.0
        self._mode = "calibrating"
        self._calibrating = True
        self._timestamp = time.time()
        self._alpha = 0.0
        self._theta = 0.0
        self._beta = 0.0

    def update(self, result: DecodeResult, timestamp: float | None = None) -> None:
        with self._lock:
            self._engagement = result.engagement
            self._load = result.load
            self._mode = result.mode
            self._calibrating = result.calibrating
            self._timestamp = timestamp if timestamp is not None else time.time()
            self._alpha = result.alpha
            self._theta = result.theta
            self._beta = result.beta

    def snapshot(self) -> BrainSnapshot:
        with self._lock:
            return BrainSnapshot(
                engagement=self._engagement,
                load=self._load,
                mode=self._mode,
                calibrating=self._calibrating,
                timestamp=self._timestamp,
                alpha=self._alpha,
                theta=self._theta,
                beta=self._beta,
            )

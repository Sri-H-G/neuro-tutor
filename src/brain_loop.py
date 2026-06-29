"""Brain loop: replay EEG through StateDecoder and update shared state."""

import time
from pathlib import Path

import mne
import numpy as np

from src.config import (
    DATA_DIR,
    DECODER_STEP_SEC,
    DECODER_WINDOW_SEC,
    FRONTAL_CHANNELS,
)
from src.decoder import StateDecoder
from src.shared_state import BrainState
from src.signal_utils import clean, select_channel_data
from src.sources import load_raw


def _iter_concat_windows(
    rest_path: Path,
    task_path: Path,
) -> tuple[float, np.ndarray, float]:
    """Yield (timestamp_sec, frontal_window, sfreq) from rest then task."""
    rest_raw = clean(load_raw(rest_path))
    task_raw = clean(load_raw(task_path))

    combined = mne.io.RawArray(
        np.concatenate([rest_raw.get_data(), task_raw.get_data()], axis=1),
        rest_raw.info,
        verbose=False,
    )

    sfreq = combined.info["sfreq"]
    window_samples = int(DECODER_WINDOW_SEC * sfreq)
    step_samples = int(DECODER_STEP_SEC * sfreq)
    data = combined.get_data()

    for start in range(0, data.shape[1] - window_samples + 1, step_samples):
        chunk = data[:, start : start + window_samples]
        frontal = select_channel_data(combined, chunk, FRONTAL_CHANNELS)
        yield start / sfreq, frontal, sfreq


def run_replay_loop(
    brain_state: BrainState,
    decoder: StateDecoder,
    rest_path: Path | None = None,
    task_path: Path | None = None,
    speed: float = 1.0,
    stop_event=None,
) -> None:
    """
    Replay rest→task EEG, updating brain_state at decoder rate.

    speed: time multiplier (20 = 20× faster than real time).
    stop_event: optional threading.Event — set to stop the loop.
    """
    rest_path = rest_path or DATA_DIR / "Subject00_1.edf"
    task_path = task_path or DATA_DIR / "Subject00_2.edf"
    sleep_sec = DECODER_STEP_SEC / speed

    for t_sec, window, sfreq in _iter_concat_windows(rest_path, task_path):
        if stop_event is not None and stop_event.is_set():
            break

        result = decoder.update(window, sfreq)
        brain_state.update(result, timestamp=t_sec)

        if sleep_sec > 0:
            time.sleep(sleep_sec)

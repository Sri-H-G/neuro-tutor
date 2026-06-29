"""Replay cleaned EEG through StateDecoder and print mode timeline."""

from collections import Counter
from pathlib import Path

import mne
import numpy as np

from src.config import (
    DATA_DIR,
    DECODER_STEP_SEC,
    DECODER_WINDOW_SEC,
    FRONTAL_CHANNELS,
)
from src.decoder import DecodeResult, StateDecoder
from src.signal_utils import clean, select_channel_data
from src.sources import load_raw


def frontal_windows(
    raw: mne.io.Raw,
    window_sec: float = DECODER_WINDOW_SEC,
    step_sec: float = DECODER_STEP_SEC,
) -> list[tuple[float, np.ndarray]]:
    """Yield (timestamp_sec, frontal_window) slices from a cleaned Raw."""
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    windows: list[tuple[float, np.ndarray]] = []
    for start in range(0, data.shape[1] - window_samples + 1, step_samples):
        chunk = data[:, start : start + window_samples]
        frontal = select_channel_data(raw, chunk, FRONTAL_CHANNELS)
        t_sec = start / sfreq
        windows.append((t_sec, frontal))
    return windows


def replay_file(
    path: Path,
    decoder: StateDecoder,
    label: str,
    print_every: int = 4,
) -> list[DecodeResult]:
    """Run decoder on one file; return all results."""
    raw = clean(load_raw(path))
    sfreq = raw.info["sfreq"]
    results: list[DecodeResult] = []

    print(f"\n--- Replay: {label} ({path.name}) ---")

    for i, (t_sec, window) in enumerate(frontal_windows(raw)):
        result = decoder.update(window, sfreq)
        results.append(result)

        if i % print_every == 0:
            cal = "  (calibrating)" if result.calibrating else ""
            print(
                f"  t={t_sec:5.1f}s  "
                f"eng={result.engagement:5.3f}  "
                f"load={result.load:5.3f}  "
                f"mode={result.mode}{cal}"
            )

    return results


def summarize_modes(results: list[DecodeResult], label: str) -> None:
    """Print mode distribution, skipping calibration updates."""
    post_cal = [r for r in results if not r.calibrating]
    if not post_cal:
        print(f"  {label}: no post-calibration updates")
        return

    counts = Counter(r.mode for r in post_cal)
    total = len(post_cal)
    print(f"  {label} mode summary (post-cal, n={total}):")
    for mode, count in counts.most_common():
        pct = 100 * count / total
        print(f"    {mode}: {count} ({pct:.0f}%)")


def replay_concatenated(
    rest_path: Path,
    task_path: Path,
    decoder: StateDecoder,
) -> tuple[list[DecodeResult], float]:
    """Replay rest then task on one decoder — watch mode flip mid-stream."""
    rest_raw = clean(load_raw(rest_path))
    task_raw = clean(load_raw(task_path))

    rest_dur = rest_raw.n_times / rest_raw.info["sfreq"]
    combined = mne.io.RawArray(
        np.concatenate([rest_raw.get_data(), task_raw.get_data()], axis=1),
        rest_raw.info,
        verbose=False,
    )

    sfreq = combined.info["sfreq"]
    window_samples = int(DECODER_WINDOW_SEC * sfreq)
    step_samples = int(DECODER_STEP_SEC * sfreq)
    data = combined.get_data()
    results: list[DecodeResult] = []

    print(f"\n--- Replay: REST → TASK concatenated ---")

    for i, start in enumerate(range(0, data.shape[1] - window_samples + 1, step_samples)):
        chunk = data[:, start : start + window_samples]
        frontal = select_channel_data(combined, chunk, FRONTAL_CHANNELS)
        t_sec = start / sfreq
        result = decoder.update(frontal, sfreq)
        results.append(result)

        if i % 8 == 0:
            segment = "REST" if t_sec < rest_dur else "TASK"
            print(
                f"  t={t_sec:5.1f}s [{segment}]  "
                f"eng={result.engagement:5.3f}  "
                f"load={result.load:5.3f}  "
                f"mode={result.mode}"
            )

    return results, rest_dur


def summarize_by_time(
    results: list[DecodeResult],
    rest_dur: float,
    label_rest: str,
    label_task: str,
) -> None:
    """Split concatenated results by rest/task boundary."""
    post_cal = [
        (i * DECODER_STEP_SEC, r)
        for i, r in enumerate(results)
        if not r.calibrating
    ]
    rest_segment = [r for t, r in post_cal if t < rest_dur]
    task_segment = [r for t, r in post_cal if t >= rest_dur]
    summarize_modes(rest_segment, label_rest)
    summarize_modes(task_segment, label_task)


def main() -> None:
    rest = DATA_DIR / "Subject00_1.edf"
    task = DATA_DIR / "Subject00_2.edf"

    decoder_rest = StateDecoder()
    rest_results = replay_file(rest, decoder_rest, "REST")
    summarize_modes(rest_results, "REST")

    decoder_task = StateDecoder()
    task_results = replay_file(task, decoder_task, "TASK")
    summarize_modes(task_results, "TASK")

    decoder_concat = StateDecoder()
    concat_results, rest_dur = replay_concatenated(rest, task, decoder_concat)
    summarize_by_time(
        concat_results,
        rest_dur,
        "CONCAT rest segment",
        "CONCAT task segment",
    )


if __name__ == "__main__":
    main()

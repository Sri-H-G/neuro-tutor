"""Export a full rest->task decoder replay to JSON for the static GitHub Pages demo.

GitHub Pages can only serve static files -- it can't run the MNE/Python
pipeline or hold a server-side Gemini key. So the live demo replays a
precomputed timeline instead of decoding EEG in the browser. This script
runs the same StateDecoder used by the real-time app once, offline, and
writes every window's (timestamp, engagement, load, mode, band powers) to
docs/timeline.json.
"""

import json

from src.brain_loop import _iter_concat_windows
from src.config import DATA_DIR
from src.decoder import StateDecoder

OUTPUT_PATH = "docs/timeline.json"


def build_timeline() -> list[dict]:
    rest_path = DATA_DIR / "Subject00_1.edf"
    task_path = DATA_DIR / "Subject00_2.edf"

    decoder = StateDecoder()
    timeline = []

    # Raw band power from MNE is in V^2/Hz (~1e-11 to 1e-13) -- scale to
    # pV^2/Hz (x1e12) so the values are chart-friendly and survive rounding
    # for a compact JSON file. Only relative scale matters for the display.
    POWER_SCALE = 1e12

    for t_sec, window, sfreq in _iter_concat_windows(rest_path, task_path):
        result = decoder.update(window, sfreq)
        timeline.append({
            "t": round(t_sec, 2),
            "engagement": round(result.engagement, 4),
            "load": round(result.load, 4),
            "mode": result.mode,
            "calibrating": result.calibrating,
            "alpha": round(result.alpha * POWER_SCALE, 4),
            "theta": round(result.theta * POWER_SCALE, 4),
            "beta": round(result.beta * POWER_SCALE, 4),
        })

    return timeline


def main() -> None:
    timeline = build_timeline()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(timeline, f, separators=(",", ":"))

    modes = {}
    for point in timeline:
        modes[point["mode"]] = modes.get(point["mode"], 0) + 1

    print(f"Wrote {len(timeline)} points to {OUTPUT_PATH}")
    print(f"Mode distribution: {modes}")


if __name__ == "__main__":
    main()

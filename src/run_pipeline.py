"""Main EEG processing pipeline."""

import json
from pathlib import Path

from src.config import DATA_DIR, OUTPUT_DIR, SAMPLE_RATE
from src.decoder import decode_band_powers
from src.gemini_layer import interpret_state
from src.signal_utils import compute_band_power
from src.sources import list_eeg_files, load_eeg


def run(file_path: Path | None = None) -> dict:
    """Run the full pipeline on one EEG file and return results."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if file_path is None:
        files = list_eeg_files()
        if not files:
            raise FileNotFoundError(f"No EEG files found in {DATA_DIR}")
        file_path = files[0]

    data, fs = load_eeg(file_path)
    band_powers = compute_band_power(data, fs)
    state = decode_band_powers(band_powers)

    try:
        interpretation = interpret_state(state)
    except ValueError:
        interpretation = "Gemini API key not configured. Set GEMINI_API_KEY in .env."

    result = {
        "file": str(file_path),
        "sample_rate": fs or SAMPLE_RATE,
        "dominant_band": state.dominant_band,
        "band_powers": state.band_powers,
        "labels": state.labels,
        "confidence": state.confidence,
        "interpretation": interpretation,
    }

    out_path = OUTPUT_DIR / f"{file_path.stem}_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Results written to {out_path}")
    return result


if __name__ == "__main__":
    run()

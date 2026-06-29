"""Load EEG data from local files."""

from pathlib import Path

import mne

from src.config import DATA_DIR


def list_eeg_files(directory: Path | None = None) -> list[Path]:
    """Return paths to supported EEG files in the data directory."""
    directory = directory or DATA_DIR
    extensions = {".edf", ".bdf", ".fif", ".set"}
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in extensions and p.is_file()
    )


def load_raw(path: Path) -> mne.io.Raw:
    """Load an EDF file, rename channels, and drop ECG/EKG."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

    rename = {
        ch: ch.replace("EEG ", "")
        for ch in raw.ch_names
        if ch.startswith("EEG ")
    }
    raw.rename_channels(rename)

    ecg_channels = [
        ch for ch in raw.ch_names
        if "ECG" in ch.upper() or "EKG" in ch.upper()
    ]
    if ecg_channels:
        raw.drop_channels(ecg_channels)

    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    return raw

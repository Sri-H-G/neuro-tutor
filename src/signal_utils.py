"""EEG signal cleaning, filtering, and feature extraction."""

import mne
import numpy as np
from scipy import signal

from src.config import FILTER_HIGH, FILTER_LOW, NOTCH_FREQ

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def clean(raw: mne.io.Raw) -> mne.io.Raw:
    """Bandpass 1–40 Hz, notch 50 Hz, average reference."""
    cleaned = raw.copy()
    cleaned.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH, verbose=False)
    cleaned.notch_filter(freqs=NOTCH_FREQ, verbose=False)
    cleaned.set_eeg_reference("average", projection=False, verbose=False)
    return cleaned


def welch_spectrum(
    data: np.ndarray,
    fs: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD averaged across channels. data shape: (n_channels, n_samples)."""
    if nperseg is None:
        nperseg = min(2048, data.shape[-1])
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=-1)
    mean_psd = np.mean(psd, axis=0)
    return freqs, mean_psd


def band_powers(
    data: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Compute mean band power averaged across channels."""
    if bands is None:
        bands = BANDS

    freqs, psd = welch_spectrum(data, fs)
    powers = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        powers[name] = float(np.mean(psd[mask]))
    return powers


def engagement_index(band_power: dict[str, float]) -> float:
    """Beta / (alpha + theta) — higher during focused mental tasks."""
    alpha = band_power["alpha"]
    theta = band_power["theta"]
    beta = band_power["beta"]
    return beta / (alpha + theta)


def relative_alpha(band_power: dict[str, float]) -> float:
    """Alpha power as a fraction of total band power."""
    total = sum(band_power.values())
    return band_power["alpha"] / total if total > 0 else 0.0


def relative_theta(band_power: dict[str, float]) -> float:
    """Theta power as a fraction of total band power."""
    total = sum(band_power.values())
    return band_power["theta"] / total if total > 0 else 0.0


def alpha_peak_height(
    data: np.ndarray,
    fs: float,
) -> tuple[float, float]:
    """Peak PSD value and its frequency within the alpha band (8–13 Hz)."""
    freqs, psd = welch_spectrum(data, fs)
    mask = (freqs >= 8) & (freqs <= 13)
    alpha_freqs = freqs[mask]
    alpha_psd = psd[mask]
    peak_idx = int(np.argmax(alpha_psd))
    return float(alpha_psd[peak_idx]), float(alpha_freqs[peak_idx])


def select_channel_data(
    raw: mne.io.Raw,
    data: np.ndarray,
    channels: list[str],
) -> np.ndarray:
    """Extract data rows for named channels."""
    indices = [raw.ch_names.index(ch) for ch in channels]
    return data[indices]

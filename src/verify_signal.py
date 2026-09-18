"""Verify load → clean → features: rest vs task spectral crossover."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import DATA_DIR, OUTPUT_DIR
from src.signal_utils import (
    alpha_peak_height,
    band_powers,
    clean,
    engagement_index,
    relative_alpha,
    select_channel_data,
    welch_spectrum,
)
from src.sources import load_raw

REST_FILE = DATA_DIR / "Subject00_1.edf"
TASK_FILE = DATA_DIR / "Subject00_2.edf"
WINDOW_SEC = 60.0
FRONTAL_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "Fz"]


def step_a_load(path: Path) -> None:
    """Print channel names and sampling rate."""
    raw = load_raw(path)
    print(f"\n=== Step A: Load check ({path.name}) ===")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Channel count: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names}")


def step_b_clean(path: Path):
    """Load and clean, no output."""
    raw = load_raw(path)
    return clean(raw)


def step_c_plot_spectra(rest_raw, task_raw) -> Path:
    """Plot Welch spectra: rest vs task."""
    fs = rest_raw.info["sfreq"]

    rest_data = rest_raw.get_data()
    task_data = task_raw.get_data()

    # Use same window length for fair comparison
    n_samples = int(WINDOW_SEC * fs)
    rest_data = rest_data[:, :min(n_samples, rest_data.shape[1])]
    task_data = task_data[:, :min(n_samples, task_data.shape[1])]

    rest_freqs, rest_psd = welch_spectrum(rest_data, fs)
    task_freqs, task_psd = welch_spectrum(task_data, fs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rest_freqs, rest_psd, label="Rest (_1)", color="#6c8cff", linewidth=1.5)
    ax.plot(task_freqs, task_psd, label="Task (_2)", color="#ff6c8c", linewidth=1.5)

    ax.axvspan(8, 13, alpha=0.12, color="#6c8cff", label="Alpha (8 to 13 Hz)")
    ax.axvspan(13, 30, alpha=0.12, color="#ff6c8c", label="Beta (13 to 30 Hz)")

    ax.set_xlim(0, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Welch PSD: Rest vs Mental Arithmetic Task (Subject00)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "spectrum_rest_vs_task.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n=== Step C: Spectrum plot saved to {out_path} ===")
    return out_path


def step_d_features(rest_raw, task_raw) -> None:
    """Compare band powers and candidate state features: all channels vs frontal."""
    fs = rest_raw.info["sfreq"]
    n_samples = int(WINDOW_SEC * fs)

    rest_all = rest_raw.get_data()[:, :n_samples]
    task_all = task_raw.get_data()[:, :min(n_samples, task_raw.n_times)]

    # Match window length to shortest recording
    n_samples = min(rest_all.shape[1], task_all.shape[1])
    rest_all = rest_all[:, :n_samples]
    task_all = task_all[:, :n_samples]

    rest_frontal = select_channel_data(rest_raw, rest_all, FRONTAL_CHANNELS)
    task_frontal = select_channel_data(task_raw, task_all, FRONTAL_CHANNELS)

    for label, rest_data, task_data in [
        ("ALL CHANNELS (20)", rest_all, task_all),
        ("FRONTAL ONLY (Fp1,Fp2,F3,F4,Fz)", rest_frontal, task_frontal),
    ]:
        rest_bp = band_powers(rest_data, fs)
        task_bp = band_powers(task_data, fs)

        rest_ei = engagement_index(rest_bp)
        task_ei = engagement_index(task_bp)
        rest_ra = relative_alpha(rest_bp)
        task_ra = relative_alpha(task_bp)
        rest_peak, rest_peak_hz = alpha_peak_height(rest_data, fs)
        task_peak, task_peak_hz = alpha_peak_height(task_data, fs)
        rest_beta_alpha = rest_bp["beta"] / rest_bp["alpha"]
        task_beta_alpha = task_bp["beta"] / task_bp["alpha"]

        print(f"\n=== Step D: {label} ===")
        print(f"Window: {n_samples / fs:.1f}s")
        print(f"\n{'Band':<8} {'Rest':>14} {'Task':>14} {'Δ (task-rest)':>14}")
        print("-" * 54)
        for band in rest_bp:
            delta = task_bp[band] - rest_bp[band]
            print(f"{band:<8} {rest_bp[band]:>14.4e} {task_bp[band]:>14.4e} {delta:>+14.4e}")

        print(f"\n{'Feature':<28} {'Rest':>12} {'Task':>12} {'Direction':>12}")
        print("-" * 56)

        features = [
            ("engagement_index β/(α+θ)", rest_ei, task_ei),
            ("relative_alpha α/total", rest_ra, task_ra),
            ("alpha_peak_height", rest_peak, task_peak),
            ("alpha_peak_freq (Hz)", rest_peak_hz, task_peak_hz),
            ("beta/alpha ratio", rest_beta_alpha, task_beta_alpha),
        ]
        for name, rest_val, task_val in features:
            if task_val > rest_val:
                direction = "task > rest"
            elif task_val < rest_val:
                direction = "rest > task"
            else:
                direction = "equal"
            print(f"{name:<28} {rest_val:>12.4e} {task_val:>12.4e} {direction:>12}")


def main() -> None:
    step_a_load(REST_FILE)

    rest_clean = step_b_clean(REST_FILE)
    task_clean = step_b_clean(TASK_FILE)
    print("\n=== Step B: Cleaned both files (bandpass 1 to 40, notch 50, avg ref) ===")

    step_c_plot_spectra(rest_clean, task_clean)
    step_d_features(rest_clean, task_clean)


if __name__ == "__main__":
    main()

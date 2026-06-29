"""Decode EEG windows into calibrated engagement, load, and mode labels."""

from dataclasses import dataclass

import numpy as np

from src.signal_utils import band_powers, relative_alpha, relative_theta

FRONTAL_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "Fz"]

MODE_FOCUSED = "focused"
MODE_OVERLOADED = "overloaded"
MODE_DISENGAGED = "disengaged"
MODE_CALIBRATING = "calibrating"


@dataclass
class DecodeResult:
    """Output of a single decoder update."""

    engagement: float
    load: float
    mode: str
    calibrating: bool


class StateDecoder:
    """
    Turn EEG windows into two calibrated 0–1 signals plus a mode label.

    Pipeline: extract → calibrate → smooth → classify.

    Calibration assumes the user starts in a relatively neutral/baseline state
    (e.g. rest before task). The first N updates collect min/max per signal;
    after that values are normalized to [0, 1] and EMA-smoothed.
    """

    def __init__(
        self,
        calibration_samples: int = 40,
        ema_alpha: float = 0.2,
        mode_threshold: float = 0.5,
        hysteresis: int = 3,
    ) -> None:
        self.calibration_samples = calibration_samples
        self.ema_alpha = ema_alpha
        self.mode_threshold = mode_threshold
        self.hysteresis = hysteresis

        self._cal_engagement: list[float] = []
        self._cal_load: list[float] = []
        self._eng_min: float | None = None
        self._eng_max: float | None = None
        self._load_min: float | None = None
        self._load_max: float | None = None

        self._smoothed_engagement: float | None = None
        self._smoothed_load: float | None = None
        self._current_mode: str = MODE_CALIBRATING
        self._pending_mode: str | None = None
        self._pending_count: int = 0

    def reset(self) -> None:
        """Clear all internal state for a fresh session."""
        self._cal_engagement.clear()
        self._cal_load.clear()
        self._eng_min = None
        self._eng_max = None
        self._load_min = None
        self._load_max = None
        self._smoothed_engagement = None
        self._smoothed_load = None
        self._current_mode = MODE_CALIBRATING
        self._pending_mode = None
        self._pending_count = 0

    def _extract_raw_signals(self, window: np.ndarray, sfreq: float) -> tuple[float, float]:
        """Frontal relative alpha → inverted engagement; relative theta → load."""
        powers = band_powers(window, sfreq)
        rel_alpha = relative_alpha(powers)
        rel_theta = relative_theta(powers)
        raw_engagement = 1.0 - rel_alpha
        raw_load = rel_theta
        return raw_engagement, raw_load

    def _normalize(self, value: float, vmin: float, vmax: float) -> float:
        span = vmax - vmin
        if span < 1e-12:
            return 0.5
        return float(np.clip((value - vmin) / span, 0.0, 1.0))

    def _ema(self, new: float, prev: float | None) -> float:
        if prev is None:
            return new
        return self.ema_alpha * new + (1.0 - self.ema_alpha) * prev

    def _classify_mode(self, engagement: float, load: float) -> str:
        if engagement < self.mode_threshold:
            return MODE_DISENGAGED
        if load >= self.mode_threshold:
            return MODE_OVERLOADED
        return MODE_FOCUSED

    def _apply_hysteresis(self, candidate: str) -> str:
        if candidate == self._current_mode:
            self._pending_mode = None
            self._pending_count = 0
            return self._current_mode

        if candidate == self._pending_mode:
            self._pending_count += 1
        else:
            self._pending_mode = candidate
            self._pending_count = 1

        if self._pending_count >= self.hysteresis:
            self._current_mode = candidate
            self._pending_mode = None
            self._pending_count = 0

        return self._current_mode

    def update(self, window: np.ndarray, sfreq: float) -> DecodeResult:
        """
        Process one EEG window (frontal channels, shape n_channels × n_samples).

        Returns engagement (0–1), load (0–1), mode, and calibrating flag.
        """
        raw_eng, raw_load = self._extract_raw_signals(window, sfreq)

        # Calibration phase — collect min/max candidates from raw values
        if len(self._cal_engagement) < self.calibration_samples:
            self._cal_engagement.append(raw_eng)
            self._cal_load.append(raw_load)

            if len(self._cal_engagement) == self.calibration_samples:
                self._eng_min = min(self._cal_engagement)
                self._eng_max = max(self._cal_engagement)
                self._load_min = min(self._cal_load)
                self._load_max = max(self._cal_load)

            return DecodeResult(
                engagement=raw_eng,
                load=raw_load,
                mode=MODE_CALIBRATING,
                calibrating=True,
            )

        norm_eng = self._normalize(raw_eng, self._eng_min, self._eng_max)
        norm_load = self._normalize(raw_load, self._load_min, self._load_max)

        self._smoothed_engagement = self._ema(norm_eng, self._smoothed_engagement)
        self._smoothed_load = self._ema(norm_load, self._smoothed_load)

        candidate = self._classify_mode(
            self._smoothed_engagement, self._smoothed_load
        )

        if self._current_mode == MODE_CALIBRATING:
            self._current_mode = candidate
            self._pending_mode = None
            self._pending_count = 0
            mode = candidate
        else:
            mode = self._apply_hysteresis(candidate)

        return DecodeResult(
            engagement=self._smoothed_engagement,
            load=self._smoothed_load,
            mode=mode,
            calibrating=False,
        )

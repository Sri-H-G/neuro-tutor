"""Configuration for the EEG processing pipeline."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# EEG processing defaults
SAMPLE_RATE = 500  # Hz (EEGMAT dataset)
FILTER_LOW = 1.0   # Hz
FILTER_HIGH = 40.0 # Hz
NOTCH_FREQ = 50.0  # Hz

# Decoder / replay
FRONTAL_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "Fz"]
DECODER_WINDOW_SEC = 2.0   # seconds per feature window
DECODER_STEP_SEC = 0.25    # 4 Hz update rate

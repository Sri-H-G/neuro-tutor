# neuro-genai-loop

EEG brain-state engine and adaptive LLM tutor. Raw EEG → cleaned frontal features → calibrated engagement/load → mode label → tutor adapts pedagogy in real time.

Demonstrated on replayed PhysioNet EEGMAT data (rest vs mental arithmetic).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
```

Download EEG files into `data/` (e.g. `Subject00_1.edf` rest, `Subject00_2.edf` task from [PhysioNet EEGMAT](https://physionet.org/content/eegmat/1.0.0/)).

## Run the web app

```bash
python -m src.run_app
```

Open **http://127.0.0.1:8080** — live brain state + adaptive tutor chat.

Options: `--speed 30`, `--narrate`, `--http-port 8080`

## Other commands

```bash
python -m src.verify_signal      # signal verification (spectrum + features)
python -m src.replay_decoder     # decoder mode timeline
python -m src.run_tutor            # CLI tutor + brain replay
python -m src.run_demo             # scripted demo for recording
```

## Branches

| Branch | Stage |
|--------|--------|
| `main` | Full app (dashboard + tutor) |
| `feature/signal-pipeline` | Load, clean, verify spectra |
| `feature/state-decoder` | + StateDecoder and replay |
| `feature/adaptive-tutor` | + Gemini tutor and brain loop |

## Project structure

```
neuro-genai-loop/
├── data/            # EEG input files (not committed)
├── src/             # Pipeline, decoder, tutor, servers
├── frontend/        # Web dashboard
└── outputs/         # Generated plots (not committed)
```

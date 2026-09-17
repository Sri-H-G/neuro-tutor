# neuro-genai-loop

EEG brain-state engine and adaptive LLM tutor. Raw EEG → cleaned frontal features → calibrated engagement/load → mode label → tutor adapts pedagogy in real time.

Demonstrated on replayed PhysioNet EEGMAT data (rest vs mental arithmetic).

**Live demo:** https://sri-h-g.github.io/neuro-tutor/ — a static replay of a
real decoded EEG session (same signal pipeline, same mode→prompt logic) with
simulated tutor replies, since a public static page can't hold a real API
key. For the full experience with live EEG replay and real Gemini responses,
run it locally (below).

## AI layer

The tutor calls the **Gemini API** (`gemini-2.0-flash` by default,
[src/gemini_layer.py](src/gemini_layer.py)) and rebuilds its system prompt on
every turn from the current decoded brain state
([src/mode_prompts.py](src/mode_prompts.py), [src/tutor.py](src/tutor.py)):
focused → advance and go deeper, overloaded → slow down and simplify,
disengaged → suggest a break. Gemini was chosen because it has a genuinely
free tier, which matters for a demo people can try without adding billing.

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
python -m src.export_timeline      # regenerate docs/timeline.json for the GitHub Pages demo
```

## GitHub Pages demo

`docs/` is a self-contained static site: it fetches `docs/timeline.json` (a
full rest→task decoder run, precomputed by `src/export_timeline.py`) and
replays it on a loop, driving the same brain-state UI as the live app. The
tutor panel picks from pre-written replies keyed by mode instead of calling
Gemini, since GitHub Pages has no backend to hold an API key — see the
banner on the page itself.

To (re)publish: regenerate `docs/timeline.json` after any decoder change,
commit it, then in the repo's **Settings → Pages** set source to *Deploy
from a branch* → `main` → `/docs`.

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

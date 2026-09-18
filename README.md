# Neuro GenAI Loop

An EEG powered adaptive tutor. It reads a person's brain activity, estimates how engaged they are and how mentally loaded they are, and uses that to change what an AI tutor teaches and how it teaches it, in real time.

**Live demo (no setup required):** https://sri-h-g.github.io/neuro-tutor/

## The idea in one paragraph

A learner's EEG is recorded while they rest and then while they do a mental task. The signal is cleaned, turned into two scores between 0 and 1 (engagement and cognitive load), and those scores are classified into a mode: focused, overloaded, or disengaged. That mode is injected into the system prompt sent to an LLM tutor before every reply, so the same lesson gets taught differently depending on how the learner's brain is actually responding, not just what they type.

## What this project demonstrates

- **Signal processing:** loading, filtering, and referencing raw EEG (MNE), then extracting frequency band power (theta, alpha, beta) from frontal channels.
- **Applied calibration:** turning noisy raw signal into two stable, session calibrated 0 to 1 scores, with smoothing and hysteresis so the classified mode doesn't flicker.
- **AI integration:** a multi turn tutor that calls the Gemini API and rebuilds its own system prompt every turn based on live sensor state, not just conversation history.
- **Full stack build:** a Python backend (WebSocket plus HTTP server) serving a real time dashboard, and a separate static site (vanilla HTML, CSS, and JavaScript, no framework) deployed to GitHub Pages.

## How it works

1. Load a rest recording and a task recording (PhysioNet EEGMAT dataset).
2. Clean the signal: bandpass filter, notch filter, average reference.
3. Slide a window across the recording and compute band power per window.
4. Calibrate the first few seconds into a baseline, then normalize every new window against it to get an engagement score and a load score.
5. Classify a mode from those two scores.
6. Before every tutor reply, read the current mode and adjust the system prompt: move faster when focused, slow down and simplify when overloaded, offer a break when disengaged.

## Tech stack

Python, MNE, NumPy, SciPy for the EEG pipeline. A lightweight WebSocket and HTTP server (standard library plus `websockets`) for the live app. Google's Gemini API for the tutor. Plain HTML, CSS, and JavaScript for both the live dashboard and the static demo, no build tooling required.

## Try it

**Live demo:** https://sri-h-g.github.io/neuro-tutor/ replays a real decoded EEG session through the same dashboard. Its tutor replies are pre written per mode instead of live API calls, since a public static page can't safely hold an API key. Everything upstream of the reply (signal cleaning, calibration, mode classification) is the real pipeline, precomputed once and replayed.

**Full local version**, with live EEG replay and real Gemini responses:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
```

Download EEG files into `data/` (`Subject00_1.edf` for rest, `Subject00_2.edf` for task, from [PhysioNet EEGMAT](https://physionet.org/content/eegmat/1.0.0/)).

```bash
python -m src.run_app
```

Open http://127.0.0.1:8080 for the live brain state view and adaptive tutor chat.

Useful flags: `--speed 30` (replay speed multiplier), `--narrate` (tutor briefly explains why it's adapting), `--http-port 8080`.

## Other commands

```bash
python -m src.verify_signal      # signal verification: spectrum and features
python -m src.replay_decoder     # print the decoder's mode timeline
python -m src.run_tutor          # CLI tutor plus brain replay
python -m src.run_demo           # scripted demo, useful for recording a video
python -m src.export_timeline    # regenerate docs/timeline.json for the GitHub Pages demo
```

## Publishing the GitHub Pages demo

`docs/` is a self contained static site. It fetches `docs/timeline.json` (a full rest to task decoder run, precomputed by `src/export_timeline.py`) and replays it on a loop through the same dashboard code as the live app.

To update it: regenerate `docs/timeline.json` after any decoder change, commit it, then in the repo's Settings, Pages, set source to "Deploy from a branch", branch `main`, folder `/docs`.

## Project structure

```
neuro-genai-loop/
├── data/            EEG input files, not committed
├── src/             signal pipeline, decoder, tutor, servers
├── frontend/        web dashboard for the live app
├── docs/            static GitHub Pages demo
└── outputs/         generated plots, not committed
```

"""Scripted demo: replay rest→task while tutor adapts to mode changes."""

import argparse
import threading
import time

from src.brain_loop import run_replay_loop
from src.decoder import StateDecoder
from src.shared_state import BrainState
from src.tutor import AdaptiveTutor


DEMO_PROMPTS = [
    "I'm ready to learn. Please start the lesson.",
    "Got it. What's the next concept?",
    "Can you go deeper on that?",
    "Let's keep going.",
    "What should we cover next?",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scripted adaptive tutor demo (for recording)"
    )
    parser.add_argument("--speed", type=float, default=30.0)
    parser.add_argument("--narrate", action="store_true", default=True)
    parser.add_argument(
        "--interval",
        type=float,
        default=45.0,
        help="Seconds between tutor prompts (wall clock)",
    )
    args = parser.parse_args()

    brain_state = BrainState()
    decoder = StateDecoder()
    tutor = AdaptiveTutor(brain_state, narrate=args.narrate)
    stop_event = threading.Event()

    thread = threading.Thread(
        target=run_replay_loop,
        args=(brain_state, decoder, None, None, args.speed, stop_event),
        daemon=True,
    )
    thread.start()

    print("=" * 60)
    print("ADAPTIVE TUTOR DEMO")
    print(
        "A tutor that adapts to a heuristic, session-calibrated estimate "
        "of cognitive state, demonstrated on replayed EEG."
    )
    print(f"Replay at {args.speed}× | prompts every {args.interval}s")
    print("=" * 60)

    for i, prompt in enumerate(DEMO_PROMPTS):
        if stop_event.is_set():
            break

        snap = brain_state.snapshot()
        print(
            f"\n--- Prompt {i + 1} | replay t≈{snap.timestamp:.0f}s | "
            f"mode={snap.mode} eng={snap.engagement:.2f} load={snap.load:.2f} ---"
        )
        print(f"You: {prompt}")

        try:
            reply = tutor.respond(prompt)
        except ValueError as exc:
            print(f"\n{exc}")
            print("Set GEMINI_API_KEY and re-run.")
            stop_event.set()
            break

        print(f"Tutor: {reply}")

        if i < len(DEMO_PROMPTS) - 1:
            time.sleep(args.interval)

    stop_event.set()
    print("\nDemo complete.")


if __name__ == "__main__":
    main()

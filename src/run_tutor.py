"""Interactive tutor chat with live brain-state replay in the background."""

import argparse
import threading

from src.brain_loop import run_replay_loop
from src.decoder import StateDecoder
from src.shared_state import BrainState
from src.tutor import AdaptiveTutor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive EEG tutor — brain replay + interactive chat"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier (default: 1.0 = real time)",
    )
    parser.add_argument(
        "--narrate",
        action="store_true",
        help="Tutor occasionally explains why it's adapting",
    )
    parser.add_argument(
        "--no-brain",
        action="store_true",
        help="Tutor only — skip brain replay (mode stays calibrating)",
    )
    args = parser.parse_args()

    brain_state = BrainState()
    decoder = StateDecoder()
    tutor = AdaptiveTutor(brain_state, narrate=args.narrate)
    stop_event = threading.Event()

    if not args.no_brain:
        thread = threading.Thread(
            target=run_replay_loop,
            args=(brain_state, decoder, None, None, args.speed, stop_event),
            daemon=True,
        )
        thread.start()
        print(
            f"Brain loop running (rest→task replay at {args.speed}× speed). "
            "Mode updates in the background."
        )
    else:
        print("Brain loop disabled — plain tutor mode.")

    print("Type your message (or 'quit' to exit).\n")

    try:
        first = tutor.start_lesson()
        snap = brain_state.snapshot()
        print(f"[mode={snap.mode} eng={snap.engagement:.2f} load={snap.load:.2f}]")
        print(f"Tutor: {first}\n")

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                break

            reply = tutor.respond(user_input)
            snap = brain_state.snapshot()
            print(
                f"[mode={snap.mode} eng={snap.engagement:.2f} "
                f"load={snap.load:.2f}]"
            )
            print(f"Tutor: {reply}\n")
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()

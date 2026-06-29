"""Plain multi-turn tutor chat — no brain state (Step 2 smoke test)."""

from src.tutor import AdaptiveTutor
from src.shared_state import BrainState


def main() -> None:
    tutor = AdaptiveTutor(BrainState(), narrate=False)
    print("Plain tutor chat (no brain loop). Type 'quit' to exit.\n")

    print(f"Tutor: {tutor.start_lesson()}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        print(f"Tutor: {tutor.respond(user_input)}\n")


if __name__ == "__main__":
    main()

"""Mode-adaptive LLM tutor — behavior bends to brain state."""

from src.gemini_layer import chat
from src.lesson import LESSON_OUTLINE_TEXT, LESSON_TOPIC
from src.mode_prompts import MODE_INSTRUCTIONS, NARRATE_INSTRUCTION
from src.shared_state import BrainState


class AdaptiveTutor:
    """
    Multi-turn tutor that reads the latest brain mode only when responding.

    Pacing: mode is sampled at response time, not on every 0.25s decoder tick.
    Hysteresis in the decoder already reduces flicker; this avoids a twitchy tutor.
    """

    def __init__(
        self,
        brain_state: BrainState,
        narrate: bool = False,
    ) -> None:
        self.brain_state = brain_state
        self.narrate = narrate
        self.history: list[dict[str, str]] = []

    def _build_system_prompt(self) -> str:
        snap = self.brain_state.snapshot()
        mode_instruction = MODE_INSTRUCTIONS.get(
            snap.mode, MODE_INSTRUCTIONS["calibrating"]
        )

        parts = [
            f"You are a concise, friendly tutor teaching: {LESSON_TOPIC}.",
            "Keep responses to 2–4 short paragraphs. Use plain language.",
            "Follow the lesson outline in order unless the learner asks to skip.",
            mode_instruction,
            f"Current signals (session-calibrated estimates): "
            f"engagement={snap.engagement:.2f}, load={snap.load:.2f}.",
            "LESSON OUTLINE:\n" + LESSON_OUTLINE_TEXT,
        ]

        if self.narrate:
            parts.append(NARRATE_INSTRUCTION)

        return "\n\n".join(parts)

    def respond(self, user_message: str) -> str:
        """Generate a tutor reply using the current brain mode."""
        system_prompt = self._build_system_prompt()
        reply = chat(system_prompt, self.history, user_message)

        self.history.append({"role": "user", "text": user_message})
        self.history.append({"role": "model", "text": reply})
        return reply

    def start_lesson(self) -> str:
        return self.respond("I'm ready to learn. Please start the lesson.")

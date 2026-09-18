"""Mode → system-prompt injection blocks for the adaptive tutor."""

MODE_FOCUSED = "focused"
MODE_OVERLOADED = "overloaded"
MODE_DISENGAGED = "disengaged"
MODE_CALIBRATING = "calibrating"

MODE_INSTRUCTIONS: dict[str, str] = {
    MODE_FOCUSED: (
        "BRAIN STATE: focused. The learner is engaged and coping well. "
        "Move forward: introduce the next concept, go a bit deeper, or quiz them."
    ),
    MODE_OVERLOADED: (
        "BRAIN STATE: overloaded. The learner is on-task but straining. "
        "Slow down: break the last idea into one smaller step, use a concrete "
        "example, and check understanding before continuing."
    ),
    MODE_DISENGAGED: (
        "BRAIN STATE: disengaged. The learner's attention is dropping. "
        "Re-engage: pause the material, suggest a short break, or switch to a "
        "lighter/more interactive framing."
    ),
    MODE_CALIBRATING: (
        "BRAIN STATE: calibrating. The EEG baseline is still being established. "
        "Teach at a steady, moderate pace until calibration completes."
    ),
}

NARRATE_INSTRUCTION = (
    "Occasionally (not every turn), briefly explain why you're adapting your "
    "teaching style in one short sentence, e.g. 'You seem to be concentrating "
    "hard, let me slow down.' Only when it feels natural."
)

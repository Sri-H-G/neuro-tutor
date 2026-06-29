"""Hardcoded lesson outline for the adaptive tutor demo."""

LESSON_TOPIC = "Introduction to Neural Networks"

LESSON_OUTLINE = [
    "What is a neuron: inputs, weights, activation, and output.",
    "Layers: how stacked neurons form a network.",
    "Forward pass: how data flows from input to prediction.",
    "Loss: measuring how wrong the prediction is.",
    "Backpropagation: how the network learns from errors.",
    "A tiny example: recognizing handwritten digits (MNIST intuition).",
]

LESSON_OUTLINE_TEXT = "\n".join(
    f"{i + 1}. {section}" for i, section in enumerate(LESSON_OUTLINE)
)

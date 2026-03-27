"""
emotion.py — Emotion detection using SamLowe/roberta-base-go_emotions-onnx

Uses the INT8 quantized ONNX model via Hugging Face Optimum for:
  - Fast inference (~5x vs PyTorch)
  - Small footprint (125MB)
  - No heavy PyTorch dependency needed
"""

from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification

MODEL_ID = "SamLowe/roberta-base-go_emotions-onnx"

_classifier = None  # lazy singleton


def load_classifier():
    """Load and cache the ONNX classifier (called once on first use)."""
    global _classifier
    if _classifier is None:
        print("[emotion] Loading ONNX model...")
        model = ORTModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            subfolder="onnx",
            file_name="model_quantized.onnx"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _classifier = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            function_to_apply="sigmoid",
        )
        print("[emotion] Model loaded.")
    return _classifier


def classify(text: str) -> list[dict]:
    """
    Run emotion classification on a single text string.

    Returns:
        List of {"label": str, "score": float} dicts for all 28 emotions,
        sorted by score descending.
    """
    clf = load_classifier()
    outputs = clf([text])[0]
    return sorted(outputs, key=lambda x: x["score"], reverse=True)


def top_emotions(text: str, top_k: int = 5, threshold: float = 0.10) -> list[dict]:
    """
    Return top-k emotions above threshold, sorted by score.
    Useful for display / debugging.
    """
    all_emotions = classify(text)
    return [e for e in all_emotions if e["score"] >= threshold][:top_k]
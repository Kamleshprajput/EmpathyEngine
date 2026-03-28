"""
emotion.py — Emotion detection using SamLowe/roberta-base-go_emotions-onnx

Uses onnxruntime directly (no optimum, no PyTorch pipeline overhead) for
minimal memory usage on constrained environments like Render's free tier.
"""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL_ID   = "SamLowe/roberta-base-go_emotions-onnx"
ONNX_FILE  = "onnx/model_quantized.onnx"

# 28 GoEmotions labels (order matches model outputs)
LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral",
]

_session   = None
_tokenizer = None


def load_classifier():
    global _session, _tokenizer
    if _session is None:
        print("[emotion] Loading ONNX model (onnxruntime direct)...")

        model_path = hf_hub_download(repo_id=MODEL_ID, filename=ONNX_FILE)
        tok_path   = hf_hub_download(repo_id=MODEL_ID, filename="tokenizer.json")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1   # limit CPU threads → less memory
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        _session   = ort.InferenceSession(model_path, sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        _tokenizer = Tokenizer.from_file(tok_path)
        _tokenizer.enable_padding(pad_id=1, pad_token="<pad>", length=128)
        _tokenizer.enable_truncation(max_length=128)

        print("[emotion] Model loaded.")
    return _session, _tokenizer


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def classify(text: str) -> list[dict]:
    """
    Run emotion classification on a single text string.

    Returns:
        List of {"label": str, "score": float} dicts for all 28 emotions,
        sorted by score descending.
    """
    session, tokenizer = load_classifier()

    enc = tokenizer.encode(text)
    input_ids      = np.array([enc.ids],           dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)

    logits = session.run(
        None,
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )[0][0]

    scores = _sigmoid(logits).tolist()
    outputs = [{"label": l, "score": s} for l, s in zip(LABELS, scores)]
    return sorted(outputs, key=lambda x: x["score"], reverse=True)


def top_emotions(text: str, top_k: int = 5, threshold: float = 0.10) -> list[dict]:
    all_emotions = classify(text)
    return [e for e in all_emotions if e["score"] >= threshold][:top_k]
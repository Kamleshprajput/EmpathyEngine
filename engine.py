"""
engine.py — Orchestrates the full Empathy Engine pipeline.

Flow:
  text → chunk → [emotion detect → blend → rules → smooth → personality] → TTS → mp3
"""

import os
import time
from emotion import classify
from voice_map import chunk_text, get_chunk_params
from tts import synthesize

OUTPUT_DIR = "outputs"


def process(text: str, output_filename: str = None) -> dict:
    """
    Full pipeline: text → emotional mp3.

    Returns:
        dict with:
          - output_path   : path to final mp3
          - chunks        : list of per-chunk analysis dicts
          - dominant      : overall dominant emotion
          - processing_ms : time taken
    """
    t0 = time.time()

    # Layer 2 — chunking
    chunks = chunk_text(text)
    print(f"[engine] {len(chunks)} chunk(s) detected")

    prev_params = None
    chunks_with_params = []
    all_emotions = []

    for i, chunk in enumerate(chunks):
        # Emotion detection per chunk
        emotion_outputs = classify(chunk)

        # Full 5-layer voice param resolution
        params = get_chunk_params(emotion_outputs, chunk, prev_params)

        # Carry forward only the numeric params for smoothing (edge-tts keys)
        prev_params = {k: params[k] for k in ("rate", "pitch", "volume", "break_ms")}

        chunks_with_params.append({"text": chunk, "params": params})
        all_emotions.append({
            "chunk": chunk,
            "dominant_emotion": params["dominant_emotion"],
            "dominant_score":   params["dominant_score"],
            "top_emotions": [
                {"label": e["label"], "score": round(e["score"], 3)}
                for e in emotion_outputs[:5]
                if e["score"] >= 0.10
            ],
            "voice_params": {
                "rate":   round(params["rate"],   1),
                "pitch":  round(params["pitch"],  1),
                "volume": round(params["volume"], 1),
            },
        })

    # Overall dominant = chunk with highest single score
    overall_dominant = max(
        all_emotions,
        key=lambda x: x["dominant_score"]
    )["dominant_emotion"]

    # Output path
    if not output_filename:
        ts = int(time.time())
        output_filename = f"output_{ts}.mp3"

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # TTS synthesis
    synthesize(chunks_with_params, output_path)

    ms = int((time.time() - t0) * 1000)
    print(f"[engine] Done in {ms}ms → {output_path}")

    return {
        "output_path":    output_path,
        "chunks":         all_emotions,
        "dominant":       overall_dominant,
        "processing_ms":  ms,
    }
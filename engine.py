"""
engine.py — Orchestrates the full Empathy Engine pipeline.

Two-pass approach:
  Pass 1 — classify all chunks, compute global rate anchor
  Pass 2 — apply full pipeline with rate anchor per chunk
"""

import os
import time
from emotion import classify
from voice_map import chunk_text, get_chunk_params, compute_global_rate_anchor
from tts import synthesize

OUTPUT_DIR = "outputs"


def process(text: str, output_filename: str = None) -> dict:
    t0 = time.time()

    chunks = chunk_text(text)
    total  = len(chunks)
    print(f"[engine] {total} chunk(s) detected")

    # ── Pass 1 — classify all chunks + collect raw rates ──────────────────
    all_emotion_outputs = []
    raw_rates           = []

    for chunk in chunks:
        emotion_outputs = classify(chunk)
        all_emotion_outputs.append(emotion_outputs)

        # Quick blend for rate estimation (no smoothing/personality yet)
        from voice_map import blend_voice_params
        blended, _, _ = blend_voice_params(emotion_outputs)
        raw_rates.append(blended["rate"])

    rate_anchor = compute_global_rate_anchor(raw_rates, max_deviation=18)
    print(f"[engine] Rate anchor: {rate_anchor:+.1f}% (deviation ±18%)")

    # ── Pass 2 — full pipeline with rate anchor ────────────────────────────
    prev_params        = None
    prev_emotion       = None
    chunks_with_params = []
    all_emotions       = []

    for i, (chunk, emotion_outputs) in enumerate(zip(chunks, all_emotion_outputs)):
        params = get_chunk_params(
            emotion_outputs, chunk, prev_params,
            prev_emotion=prev_emotion,
            position=i,
            total_chunks=total,
            rate_anchor=rate_anchor,
        )

        prev_params  = {k: params[k] for k in ("rate", "pitch", "volume", "pre_break_ms", "break_ms")}
        prev_emotion = params["dominant_emotion"]

        chunks_with_params.append({"text": chunk, "params": params})
        all_emotions.append({
            "chunk":            chunk,
            "dominant_emotion": params["dominant_emotion"],
            "dominant_score":   params["dominant_score"],
            "top_emotions": [
                {"label": e["label"], "score": round(e["score"], 3)}
                for e in emotion_outputs[:5]
                if e["score"] >= 0.10
            ],
            "voice_params": {
                "rate":         round(params["rate"],          1),
                "pitch":        round(params["pitch"],         1),
                "volume":       round(params["volume"],        1),
                "pre_break_ms": round(params["pre_break_ms"],  0),
                "break_ms":     round(params["break_ms"],      0),
            },
        })

    overall_dominant = max(
        all_emotions, key=lambda x: x["dominant_score"]
    )["dominant_emotion"]

    if not output_filename:
        output_filename = f"output_{int(time.time())}.mp3"

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    synthesize(chunks_with_params, output_path)

    ms = int((time.time() - t0) * 1000)
    print(f"[engine] Done in {ms}ms → {output_path}")

    return {
        "output_path":    output_path,
        "chunks":         all_emotions,
        "dominant":       overall_dominant,
        "processing_ms":  ms,
        "rate_anchor":    round(rate_anchor, 1),
    }
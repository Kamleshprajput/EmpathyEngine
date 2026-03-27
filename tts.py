"""
tts.py — ElevenLabs TTS integration for the Empathy Engine.

Handles:
  - Per-chunk synthesis with individual voice params
  - Audio concatenation across chunks
  - Output as a single .mp3 file
"""

import os
import io
import time
import requests
from pydub import AudioSegment

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "LGdTKIPPsViXOIz5wFyy")  # Rachel
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"

TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

# Silence padding between chunks (ms) — gives a natural breath between clauses
INTER_CHUNK_SILENCE_MS = 180


def _synthesize_chunk(text: str, params: dict, retries: int = 3) -> bytes:
    """
    Call ElevenLabs API for a single text chunk with given voice params.
    Returns raw mp3 bytes.
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability":        round(float(params["stability"]), 3),
            "similarity_boost": 0.75,  # kept fixed — personality comes from other params
            "style":            round(float(params["style"]), 3),
            "use_speaker_boost": True,
            "speed":            round(float(params["speed"]), 3),
        },
    }

    for attempt in range(retries):
        try:
            resp = requests.post(TTS_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.content
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
            else:
                raise RuntimeError(f"ElevenLabs API error after {retries} attempts: {e}")


def synthesize(chunks_with_params: list[dict], output_path: str) -> str:
    """
    Synthesize a list of chunks and stitch into a single mp3.

    Args:
        chunks_with_params: list of dicts, each with:
            - "text"      : str
            - "params"    : voice param dict (stability, style, speed, ...)
        output_path: path to save final mp3

    Returns:
        output_path
    """
    silence = AudioSegment.silent(duration=INTER_CHUNK_SILENCE_MS)
    combined = AudioSegment.empty()

    for i, item in enumerate(chunks_with_params):
        text = item["text"].strip()
        params = item["params"]

        if not text:
            continue

        emotion = params.get("dominant_emotion", "neutral")
        score = params.get("dominant_score", 0.0)
        print(
            f"[tts] Chunk {i+1}/{len(chunks_with_params)} | "
            f"emotion={emotion} ({score:.2f}) | "
            f"stability={params['stability']:.2f} style={params['style']:.2f} speed={params['speed']:.2f}"
        )

        mp3_bytes = _synthesize_chunk(text, params)
        segment = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")

        combined += segment
        if i < len(chunks_with_params) - 1:
            combined += silence

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined.export(output_path, format="mp3")
    print(f"[tts] Saved → {output_path}")
    return output_path

"""
tts.py — edge-tts synthesis for the Empathy Engine.

Synthesizes each chunk separately with its own rate/pitch/volume,
then concatenates all chunks into a single mp3 using pydub.
"""

import os
import asyncio
import tempfile
import edge_tts
from pydub import AudioSegment

VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")


async def _synthesize_chunk(text: str, rate: str, pitch: str, volume: str, output_path: str):
    """Synthesize a single chunk with given prosody params."""
    communicate = edge_tts.Communicate(
        text,
        VOICE,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )
    await communicate.save(output_path)


def _run_async(coro):
    """Run async coroutine safely from a sync/thread context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def synthesize(chunks_with_params: list[dict], output_path: str) -> str:
    """
    Synthesize each chunk individually then stitch into one mp3.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    combined = AudioSegment.empty()
    tmp_files = []

    for i, item in enumerate(chunks_with_params):
        text = item["text"].strip()
        p    = item["params"]

        # Clamp and format params for edge-tts
        rate   = max(-50, min(50,  int(round(p["rate"]))))
        pitch  = max(-50, min(50,  int(round(p["pitch"]))))
        volume = max(-50, min(50,  int(round(p["volume"]))))
        brk_ms = max(80,  int(round(p["break_ms"])))

        rate_str   = f"{rate:+d}%"
        pitch_str  = f"{pitch:+d}Hz"
        volume_str = f"{volume:+d}%"

        print(
            f"[tts] Chunk {i+1}/{len(chunks_with_params)} | "
            f"emotion={p.get('dominant_emotion','?')} ({p.get('dominant_score',0):.2f}) | "
            f"rate={rate_str} pitch={pitch_str} volume={volume_str}"
        )

        # Write each chunk to a temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        tmp_files.append(tmp.name)

        _run_async(_synthesize_chunk(text, rate_str, pitch_str, volume_str, tmp.name))

        segment = AudioSegment.from_mp3(tmp.name)
        combined += segment

        # Add silence break between chunks
        if i < len(chunks_with_params) - 1:
            combined += AudioSegment.silent(duration=brk_ms)

    # Export final stitched mp3
    combined.export(output_path, format="mp3")

    # Cleanup temp files
    for f in tmp_files:
        try:
            os.remove(f)
        except Exception:
            pass

    print(f"[tts] Saved → {output_path}")
    return output_path
"""
tts.py — edge-tts synthesis for the Empathy Engine.

Synthesizes each chunk separately with its own rate/pitch/volume,
adds pre_break silence before and post break silence after each chunk,
then concatenates everything into a single mp3 using pydub.
"""

import os
import asyncio
import tempfile
import edge_tts
from pydub import AudioSegment

VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")


async def _synthesize_chunk(text: str, rate: str, pitch: str, volume: str, output_path: str):
    communicate = edge_tts.Communicate(text, VOICE, rate=rate, pitch=pitch, volume=volume)
    await communicate.save(output_path)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _clamp(val, lo, hi):
    return max(lo, min(hi, int(round(val))))


def synthesize(chunks_with_params: list[dict], output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    combined  = AudioSegment.empty()
    tmp_files = []

    for i, item in enumerate(chunks_with_params):
        text = item["text"].strip()
        p    = item["params"]

        rate      = _clamp(p["rate"],         -45, +45)
        pitch     = _clamp(p["pitch"],         -15, +15)
        volume    = _clamp(p["volume"],        -30, +20)
        pre_brk   = max(0,   int(round(p.get("pre_break_ms", 0))))
        post_brk  = max(80,  int(round(p.get("break_ms", 200))))

        rate_str   = f"{rate:+d}%"
        pitch_str  = f"{pitch:+d}Hz"
        volume_str = f"{volume:+d}%"

        print(
            f"[tts] Chunk {i+1}/{len(chunks_with_params)} | "
            f"emotion={p.get('dominant_emotion','?')} ({p.get('dominant_score',0):.2f}) | "
            f"rate={rate_str} pitch={pitch_str} volume={volume_str} | "
            f"pre={pre_brk}ms post={post_brk}ms"
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        tmp_files.append(tmp.name)

        _run_async(_synthesize_chunk(text, rate_str, pitch_str, volume_str, tmp.name))

        segment = AudioSegment.from_mp3(tmp.name)

        # Pre-break silence (hesitation before clause)
        if pre_brk > 0:
            combined += AudioSegment.silent(duration=pre_brk)

        combined += segment

        # Post-break silence (pause after clause)
        if i < len(chunks_with_params) - 1:
            combined += AudioSegment.silent(duration=post_brk)

    combined.export(output_path, format="mp3")

    for f in tmp_files:
        try:
            os.remove(f)
        except Exception:
            pass

    print(f"[tts] Saved → {output_path}")
    return output_path
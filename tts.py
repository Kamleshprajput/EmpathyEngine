"""
tts.py — edge-tts synthesis for the Empathy Engine.

Synthesizes each chunk separately with its own rate/pitch/volume,
then concatenates using ffmpeg directly (no pydub dependency).
"""

import os
import asyncio
import tempfile
import subprocess
import edge_tts

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


def _concat_with_silence(segments: list[dict], output_path: str):
    """
    Concatenate mp3 segments with silence gaps using ffmpeg.
    segments: list of {"file": path, "pre_ms": int, "post_ms": int}
    """
    # Build a list of inputs: silence + chunk + silence
    inputs  = []
    filters = []
    idx     = 0

    for seg in segments:
        pre_ms  = max(0, seg["pre_ms"])
        post_ms = max(0, seg["post_ms"])

        if pre_ms > 0:
            # Generate silence as input
            inputs  += ["-f", "lavfi", "-t", f"{pre_ms/1000:.3f}", "-i", "anullsrc=r=24000:cl=mono"]
            filters.append(f"[{idx}]")
            idx += 1

        inputs  += ["-i", seg["file"]]
        filters.append(f"[{idx}]")
        idx += 1

        if post_ms > 0:
            inputs  += ["-f", "lavfi", "-t", f"{post_ms/1000:.3f}", "-i", "anullsrc=r=24000:cl=mono"]
            filters.append(f"[{idx}]")
            idx += 1

    filter_str = "".join(filters) + f"concat=n={len(filters)}:v=0:a=1[out]"

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + ["-filter_complex", filter_str, "-map", "[out]",
           "-c:a", "libmp3lame", "-q:a", "4", output_path]
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr}")


def synthesize(chunks_with_params: list[dict], output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    tmp_files = []
    segments  = []

    for i, item in enumerate(chunks_with_params):
        text = item["text"].strip()
        p    = item["params"]

        rate      = _clamp(p["rate"],          -45, +45)
        pitch     = _clamp(p["pitch"],          -15, +15)
        volume    = _clamp(p["volume"],         -30, +20)
        pre_ms    = max(0,  int(round(p.get("pre_break_ms", 0))))
        post_ms   = max(80, int(round(p.get("break_ms", 200)))) if i < len(chunks_with_params) - 1 else 0

        rate_str   = f"{rate:+d}%"
        pitch_str  = f"{pitch:+d}Hz"
        volume_str = f"{volume:+d}%"

        print(
            f"[tts] Chunk {i+1}/{len(chunks_with_params)} | "
            f"emotion={p.get('dominant_emotion','?')} ({p.get('dominant_score',0):.2f}) | "
            f"rate={rate_str} pitch={pitch_str} volume={volume_str} | "
            f"pre={pre_ms}ms post={post_ms}ms"
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        tmp_files.append(tmp.name)

        _run_async(_synthesize_chunk(text, rate_str, pitch_str, volume_str, tmp.name))

        segments.append({
            "file":    tmp.name,
            "pre_ms":  pre_ms,
            "post_ms": post_ms,
        })

    _concat_with_silence(segments, output_path)

    for f in tmp_files:
        try:
            os.remove(f)
        except Exception:
            pass

    print(f"[tts] Saved → {output_path}")
    return output_path
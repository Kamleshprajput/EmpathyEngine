"""
voice_map.py — The brain of the Empathy Engine.

5-layer pipeline:
  1. Emotion Blending     — weighted blend of top-3 emotions
  2. Prosody Chunking     — per-clause emotion detection
  3. Conversational Rules — punctuation / caps overrides
  4. Transition Smoothing — carry-over between chunks
  5. Personality Baseline — consistent voice character

edge-tts params:
  rate   → speech speed,  e.g. "+20%" or "-15%"  (relative to baseline)
  pitch  → tonal height,  e.g. "+10Hz" or "-8Hz"
  volume → amplitude,     e.g. "+15%" or "-20%"

SSML additions:
  <emphasis level>  → word-level stress
  <break time>      → natural pauses between clauses
"""

import re

# ---------------------------------------------------------------------------
# Voice profiles for all 28 go_emotions labels
# rate/pitch/volume are INTEGER offsets — formatted into SSML strings later
#   rate:   -30 to +30 (%)
#   pitch:  -15 to +15 (Hz)
#   volume: -30 to +20 (%)
# emphasis: "strong" | "moderate" | "reduced" | "none"
# break_ms: pause injected AFTER this chunk (ms)
# ---------------------------------------------------------------------------

VOICE_PROFILES = {
    # ── High energy positive ────────────────────────────────────────────────
    "joy":            {"rate": +22, "pitch": +10, "volume": +18, "emphasis": "strong",   "break_ms": 150},
    "excitement":     {"rate": +28, "pitch": +13, "volume": +20, "emphasis": "strong",   "break_ms": 100},
    "amusement":      {"rate": +18, "pitch": +8,  "volume": +15, "emphasis": "moderate", "break_ms": 150},
    "love":           {"rate": +8,  "pitch": +6,  "volume": +10, "emphasis": "moderate", "break_ms": 200},
    "admiration":     {"rate": +6,  "pitch": +4,  "volume": +8,  "emphasis": "moderate", "break_ms": 200},
    "gratitude":      {"rate": +4,  "pitch": +3,  "volume": +6,  "emphasis": "moderate", "break_ms": 220},
    "pride":          {"rate": +10, "pitch": +5,  "volume": +12, "emphasis": "moderate", "break_ms": 180},
    "optimism":       {"rate": +10, "pitch": +5,  "volume": +10, "emphasis": "moderate", "break_ms": 180},
    "relief":         {"rate": -5,  "pitch": +2,  "volume": -5,  "emphasis": "reduced",  "break_ms": 250},
    "approval":       {"rate": +4,  "pitch": +2,  "volume": +5,  "emphasis": "moderate", "break_ms": 200},
    "desire":         {"rate": +5,  "pitch": +4,  "volume": +6,  "emphasis": "moderate", "break_ms": 200},

    # ── Curious / inquisitive ───────────────────────────────────────────────
    "curiosity":      {"rate": +8,  "pitch": +6,  "volume": +5,  "emphasis": "moderate", "break_ms": 180},
    "confusion":      {"rate": -8,  "pitch": +4,  "volume": -5,  "emphasis": "reduced",  "break_ms": 280},
    "realization":    {"rate": -10, "pitch": +3,  "volume": -3,  "emphasis": "moderate", "break_ms": 300},
    "surprise":       {"rate": +20, "pitch": +12, "volume": +15, "emphasis": "strong",   "break_ms": 150},

    # ── Soft / warm ─────────────────────────────────────────────────────────
    "caring":         {"rate": -8,  "pitch": +2,  "volume": -8,  "emphasis": "reduced",  "break_ms": 250},

    # ── Negative / heavy ────────────────────────────────────────────────────
    "sadness":        {"rate": -20, "pitch": -8,  "volume": -18, "emphasis": "reduced",  "break_ms": 400},
    "grief":          {"rate": -28, "pitch": -12, "volume": -22, "emphasis": "reduced",  "break_ms": 500},
    "remorse":        {"rate": -18, "pitch": -7,  "volume": -15, "emphasis": "reduced",  "break_ms": 380},
    "disappointment": {"rate": -15, "pitch": -6,  "volume": -12, "emphasis": "reduced",  "break_ms": 350},
    "embarrassment":  {"rate": -12, "pitch": -4,  "volume": -10, "emphasis": "reduced",  "break_ms": 300},

    # ── Aggressive / tense ──────────────────────────────────────────────────
    "anger":          {"rate": +25, "pitch": +8,  "volume": +20, "emphasis": "strong",   "break_ms": 120},
    "annoyance":      {"rate": +15, "pitch": +5,  "volume": +12, "emphasis": "moderate", "break_ms": 150},
    "disapproval":    {"rate": +10, "pitch": +3,  "volume": +8,  "emphasis": "moderate", "break_ms": 180},
    "disgust":        {"rate": -5,  "pitch": -4,  "volume": +8,  "emphasis": "moderate", "break_ms": 200},

    # ── Anxious ─────────────────────────────────────────────────────────────
    "fear":           {"rate": +20, "pitch": +10, "volume": -8,  "emphasis": "strong",   "break_ms": 150},
    "nervousness":    {"rate": +15, "pitch": +7,  "volume": -5,  "emphasis": "moderate", "break_ms": 180},

    # ── Baseline ────────────────────────────────────────────────────────────
    "neutral":        {"rate":  0,  "pitch":  0,  "volume":  0,  "emphasis": "none",     "break_ms": 200},
}

NEUTRAL_PARAMS = {"rate": 0, "pitch": 0, "volume": 0, "emphasis": "none", "break_ms": 200}

# ---------------------------------------------------------------------------
# Personality baseline
# ---------------------------------------------------------------------------

PERSONALITY = {
    "expressiveness":  0.85,  # scales rate + pitch deltas
    "volume_boost":    0.90,  # scales volume deltas
    "pace_preference": 1.00,  # multiplier on final rate
}

# ---------------------------------------------------------------------------
# Conversational rules
# ---------------------------------------------------------------------------

CONVERSATIONAL_RULES = [
    {
        "name": "question",
        "trigger": lambda t: t.strip().endswith("?"),
        "modifier": {"rate": -5, "pitch": +4, "volume": 0},
    },
    {
        "name": "exclamation",
        "trigger": lambda t: t.strip().endswith("!"),
        "modifier": {"rate": +8, "pitch": +5, "volume": +8},
    },
    {
        "name": "trailing",
        "trigger": lambda t: t.strip().endswith("..."),
        "modifier": {"rate": -10, "pitch": -3, "volume": -8},
    },
    {
        "name": "emphasis",
        "trigger": lambda t: any(w.isupper() and len(w) > 1 for w in t.split()),
        "modifier": {"rate": +5, "pitch": +6, "volume": +10},
    },
    {
        "name": "soft_comma_pause",
        "trigger": lambda t: t.count(",") >= 2,
        "modifier": {"rate": -5, "pitch": 0, "volume": -3},
    },
]

# ---------------------------------------------------------------------------
# Layer 1 — Emotion Blending
# ---------------------------------------------------------------------------

def blend_voice_params(emotion_outputs, top_k=3, threshold=0.15):
    candidates = sorted(
        [e for e in emotion_outputs if e["score"] >= threshold],
        key=lambda x: x["score"],
        reverse=True,
    )[:top_k]

    if not candidates:
        return dict(NEUTRAL_PARAMS), "neutral", 0.5

    total = sum(e["score"] for e in candidates)
    blended = {"rate": 0.0, "pitch": 0.0, "volume": 0.0, "break_ms": 0.0}

    for e in candidates:
        weight = e["score"] / total
        profile = VOICE_PROFILES.get(e["label"], NEUTRAL_PARAMS)
        for param in ("rate", "pitch", "volume", "break_ms"):
            blended[param] += profile[param] * weight

    # Emphasis from dominant emotion
    dominant = candidates[0]
    blended["emphasis"] = VOICE_PROFILES.get(dominant["label"], NEUTRAL_PARAMS)["emphasis"]

    return blended, dominant["label"], dominant["score"]

# ---------------------------------------------------------------------------
# Layer 3 — Conversational Rules
# ---------------------------------------------------------------------------

def apply_conversational_rules(params, text):
    p = dict(params)
    for rule in CONVERSATIONAL_RULES:
        if rule["trigger"](text):
            for param, delta in rule["modifier"].items():
                if param in p and isinstance(p[param], (int, float)):
                    p[param] = p[param] + delta
    return p

# ---------------------------------------------------------------------------
# Layer 4 — Transition Smoothing
# ---------------------------------------------------------------------------

def smooth_transition(prev_params, next_params, alpha=0.25):
    if prev_params is None:
        return dict(next_params)
    smoothed = dict(next_params)
    for param in ("rate", "pitch", "volume", "break_ms"):
        smoothed[param] = (1 - alpha) * next_params[param] + alpha * prev_params[param]
    return smoothed

# ---------------------------------------------------------------------------
# Layer 5 — Personality Baseline
# ---------------------------------------------------------------------------

def apply_personality(params, emotion_label):
    p = dict(params)
    p["rate"]   = p["rate"]   * PERSONALITY["expressiveness"] * PERSONALITY["pace_preference"]
    p["pitch"]  = p["pitch"]  * PERSONALITY["expressiveness"]
    p["volume"] = p["volume"] * PERSONALITY["volume_boost"]
    return p

# ---------------------------------------------------------------------------
# Layer 2 — Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(text):
    raw = re.split(r'(?<=[.!?,;])\s+', text.strip())
    chunks = [c.strip() for c in raw if c.strip()]
    return chunks if chunks else [text.strip()]

# ---------------------------------------------------------------------------
# SSML Builder
# ---------------------------------------------------------------------------

def _clamp(val, lo, hi):
    return max(lo, min(hi, int(round(val))))

def build_ssml(chunks_with_params, voice_name="en-US-JennyNeural"):
    """
    Build a full SSML document from chunks and their voice params.
    Each chunk gets its own <prosody> block with <emphasis> and <break>.
    """
    lines = [
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"',
        '       xmlns:mstts="http://www.w3.org/2001/mstts"',
        '       xml:lang="en-US">',
        f'  <voice name="{voice_name}">',
    ]

    for item in chunks_with_params:
        text   = item["text"].strip()
        p      = item["params"]

        rate   = _clamp(p["rate"],   -30, +30)
        pitch  = _clamp(p["pitch"],  -15, +15)
        volume = _clamp(p["volume"], -30, +20)
        brk    = max(80, int(round(p["break_ms"])))
        emph   = p.get("emphasis", "none")

        rate_str   = f"{rate:+d}%"
        pitch_str  = f"{pitch:+d}Hz"
        volume_str = f"{volume:+d}%"

        inner = (
            f'<emphasis level="{emph}">{text}</emphasis>'
            if emph and emph != "none"
            else text
        )

        lines.append(
            f'    <prosody rate="{rate_str}" pitch="{pitch_str}" volume="{volume_str}">'
            f'{inner}'
            f'</prosody>'
        )
        lines.append(f'    <break time="{brk}ms"/>')

    lines += ["  </voice>", "</speak>"]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Master pipeline — get params for a single chunk
# ---------------------------------------------------------------------------

def get_chunk_params(emotion_outputs, chunk_text_str, prev_params=None):
    blended, dominant_emotion, dominant_score = blend_voice_params(emotion_outputs)
    blended = apply_conversational_rules(blended, chunk_text_str)
    blended = smooth_transition(prev_params, blended, alpha=0.25)
    blended = apply_personality(blended, dominant_emotion)
    blended["dominant_emotion"] = dominant_emotion
    blended["dominant_score"]   = round(dominant_score, 4)
    return blended
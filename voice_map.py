"""
voice_map.py — The brain of the Empathy Engine.

5-layer pipeline:
  1. Emotion Blending     — weighted blend of top-3 emotions
  2. Prosody Chunking     — per-clause emotion detection
  3. Conversational Rules — punctuation / caps overrides
  4. Transition Smoothing — carry-over between chunks
  5. Personality Baseline — consistent voice character
"""

import re

# ---------------------------------------------------------------------------
# Voice profiles for all 28 go_emotions labels
# ElevenLabs params: stability (0-1), style (0-1), speed (0.7-1.2)
# stability  → lower = more expressive/variable
# style      → higher = more dramatic/emotional delivery
# speed      → controls pace
# ---------------------------------------------------------------------------

VOICE_PROFILES = {
    # ── High energy positive ────────────────────────────────────────────────
    "joy":            {"stability": 0.28, "style": 0.88, "speed": 1.12},
    "excitement":     {"stability": 0.22, "style": 0.92, "speed": 1.18},
    "amusement":      {"stability": 0.32, "style": 0.82, "speed": 1.10},
    "love":           {"stability": 0.38, "style": 0.76, "speed": 1.04},
    "admiration":     {"stability": 0.44, "style": 0.66, "speed": 1.04},
    "gratitude":      {"stability": 0.50, "style": 0.58, "speed": 0.98},
    "pride":          {"stability": 0.38, "style": 0.72, "speed": 1.06},
    "optimism":       {"stability": 0.44, "style": 0.66, "speed": 1.05},
    "relief":         {"stability": 0.54, "style": 0.44, "speed": 0.93},
    "approval":       {"stability": 0.54, "style": 0.46, "speed": 1.00},
    "desire":         {"stability": 0.44, "style": 0.62, "speed": 1.01},

    # ── Curious / inquisitive ───────────────────────────────────────────────
    "curiosity":      {"stability": 0.38, "style": 0.62, "speed": 1.06},
    "confusion":      {"stability": 0.34, "style": 0.56, "speed": 0.94},
    "realization":    {"stability": 0.48, "style": 0.56, "speed": 0.91},
    "surprise":       {"stability": 0.22, "style": 0.82, "speed": 1.12},

    # ── Soft / warm ─────────────────────────────────────────────────────────
    "caring":         {"stability": 0.60, "style": 0.40, "speed": 0.91},

    # ── Negative / heavy ────────────────────────────────────────────────────
    "sadness":        {"stability": 0.74, "style": 0.20, "speed": 0.80},
    "grief":          {"stability": 0.80, "style": 0.14, "speed": 0.76},
    "remorse":        {"stability": 0.70, "style": 0.24, "speed": 0.84},
    "disappointment": {"stability": 0.68, "style": 0.26, "speed": 0.87},
    "embarrassment":  {"stability": 0.64, "style": 0.30, "speed": 0.89},

    # ── Aggressive / tense ──────────────────────────────────────────────────
    "anger":          {"stability": 0.18, "style": 0.94, "speed": 1.18},
    "annoyance":      {"stability": 0.28, "style": 0.76, "speed": 1.10},
    "disapproval":    {"stability": 0.34, "style": 0.66, "speed": 1.04},
    "disgust":        {"stability": 0.32, "style": 0.72, "speed": 0.89},

    # ── Anxious ─────────────────────────────────────────────────────────────
    "fear":           {"stability": 0.24, "style": 0.80, "speed": 1.14},
    "nervousness":    {"stability": 0.28, "style": 0.72, "speed": 1.09},

    # ── Baseline ────────────────────────────────────────────────────────────
    "neutral":        {"stability": 0.60, "style": 0.30, "speed": 1.00},
}

NEUTRAL_PARAMS = {"stability": 0.60, "style": 0.30, "speed": 1.00}

# ---------------------------------------------------------------------------
# Personality baseline — defines the "character" of the voice
# Tune these to adjust the overall feel of the engine
# ---------------------------------------------------------------------------

PERSONALITY = {
    "expressiveness": 0.85,   # scales style param  (0=flat, 1=full range)
    "pace_preference": 1.00,  # multiplier on speed
    "warmth": 0.55,           # nudges stability up for warm emotions
}

WARM_EMOTIONS = {
    "joy", "love", "caring", "gratitude", "relief",
    "optimism", "admiration", "approval", "desire"
}

# ---------------------------------------------------------------------------
# Conversational rules — punctuation / linguistic overrides
# ---------------------------------------------------------------------------

CONVERSATIONAL_RULES = [
    {
        "name": "question",
        "trigger": lambda t: t.strip().endswith("?"),
        "modifier": {"stability": -0.05, "style": +0.10, "speed": -0.04},
    },
    {
        "name": "exclamation",
        "trigger": lambda t: t.strip().endswith("!"),
        "modifier": {"stability": -0.08, "style": +0.12, "speed": +0.05},
    },
    {
        "name": "trailing",
        "trigger": lambda t: t.strip().endswith("..."),
        "modifier": {"stability": +0.10, "style": -0.10, "speed": -0.08},
    },
    {
        "name": "emphasis",
        "trigger": lambda t: any(w.isupper() and len(w) > 1 for w in t.split()),
        "modifier": {"stability": -0.10, "style": +0.15, "speed": +0.03},
    },
    {
        "name": "soft_comma_pause",
        "trigger": lambda t: t.count(",") >= 2,
        "modifier": {"stability": +0.05, "style": -0.05, "speed": -0.03},
    },
]

# ---------------------------------------------------------------------------
# Layer 1 — Emotion Blending
# ---------------------------------------------------------------------------

def blend_voice_params(emotion_outputs, top_k=3, threshold=0.15):
    """
    Weighted blend of top-k emotions above threshold.
    Returns (blended_params dict, dominant_emotion str, dominant_score float).
    """
    candidates = sorted(
        [e for e in emotion_outputs if e["score"] >= threshold],
        key=lambda x: x["score"],
        reverse=True,
    )[:top_k]

    if not candidates:
        return dict(NEUTRAL_PARAMS), "neutral", 0.5

    total = sum(e["score"] for e in candidates)
    blended = {"stability": 0.0, "style": 0.0, "speed": 0.0}

    for e in candidates:
        weight = e["score"] / total
        profile = VOICE_PROFILES.get(e["label"], NEUTRAL_PARAMS)
        for param in blended:
            blended[param] += profile[param] * weight

    dominant = candidates[0]
    return blended, dominant["label"], dominant["score"]

# ---------------------------------------------------------------------------
# Layer 3 — Conversational Rules
# ---------------------------------------------------------------------------

def apply_conversational_rules(params, text):
    """Nudge params based on punctuation and linguistic patterns."""
    p = dict(params)
    for rule in CONVERSATIONAL_RULES:
        if rule["trigger"](text):
            for param, delta in rule["modifier"].items():
                p[param] = round(max(0.0, min(1.0, p[param] + delta)), 4)
    return p

# ---------------------------------------------------------------------------
# Layer 4 — Transition Smoothing
# ---------------------------------------------------------------------------

def smooth_transition(prev_params, next_params, alpha=0.25):
    """
    Blend previous chunk state into next.
    alpha=0 → instant switch, alpha=1 → never changes.
    """
    if prev_params is None:
        return dict(next_params)
    return {
        param: round((1 - alpha) * next_params[param] + alpha * prev_params[param], 4)
        for param in next_params
    }

# ---------------------------------------------------------------------------
# Layer 5 — Personality Baseline
# ---------------------------------------------------------------------------

def apply_personality(params, emotion_label):
    """Apply global voice character on top of per-emotion params."""
    p = dict(params)
    p["style"] = round(min(1.0, p["style"] * PERSONALITY["expressiveness"]), 4)
    p["speed"] = round(min(1.2, max(0.7, p["speed"] * PERSONALITY["pace_preference"])), 4)
    if emotion_label in WARM_EMOTIONS:
        p["stability"] = round(min(1.0, p["stability"] + PERSONALITY["warmth"] * 0.08), 4)
    return p

# ---------------------------------------------------------------------------
# Layer 2 — Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(text):
    """
    Split text into prosodic chunks at natural pause boundaries.
    Preserves trailing punctuation in each chunk.
    """
    raw = re.split(r'(?<=[.!?,;])\s+', text.strip())
    chunks = [c.strip() for c in raw if c.strip()]
    return chunks if chunks else [text.strip()]

# ---------------------------------------------------------------------------
# Master pipeline — get params for a single chunk given classifier output
# ---------------------------------------------------------------------------

def get_chunk_params(emotion_outputs, chunk_text_str, prev_params=None):
    """
    Full 5-layer pipeline for one text chunk.

    Args:
        emotion_outputs : raw list of {label, score} dicts from ONNX model
        chunk_text_str  : the chunk text (for rule application)
        prev_params     : params from previous chunk (for smoothing), or None

    Returns:
        dict with stability, style, speed + dominant_emotion, dominant_score
    """
    # Layer 1 — blend
    blended, dominant_emotion, dominant_score = blend_voice_params(emotion_outputs)

    # Layer 3 — conversational rules
    blended = apply_conversational_rules(blended, chunk_text_str)

    # Layer 4 — smoothing
    blended = smooth_transition(prev_params, blended, alpha=0.25)

    # Layer 5 — personality
    blended = apply_personality(blended, dominant_emotion)

    blended["dominant_emotion"] = dominant_emotion
    blended["dominant_score"] = round(dominant_score, 4)

    return blended

"""
voice_map.py — The brain of the Empathy Engine.

5-layer pipeline:
  1. Emotion Blending       — weighted blend of top-3 emotions
  2. Prosody Chunking       — per-clause emotion detection
  3. Conversational Rules   — punctuation / caps overrides
  4. Transition Smoothing   — dynamic alpha based on emotion distance
  5. Personality Baseline   — consistent voice character

Additional:
  - Recalibrated rates (base capped at ±30, amplifier handles extremes)
  - Better pitch/volume separation at emotional poles
  - pre_break_ms — hesitation before clause
  - Transition pause (0.5–0.8s) on sudden emotion pole changes
  - Sentence position awareness
"""

import re

# ---------------------------------------------------------------------------
# Emotion poles — for detecting sudden changes
# ---------------------------------------------------------------------------

EMOTION_POLES = {
    "positive":   {"joy", "excitement", "amusement", "love", "gratitude",
                   "pride", "optimism", "relief", "admiration", "approval", "desire"},
    "negative":   {"sadness", "grief", "remorse", "disappointment",
                   "embarrassment", "disgust"},
    "aggressive": {"anger", "annoyance", "disapproval"},
    "anxious":    {"fear", "nervousness"},
    "neutral":    {"neutral", "realization", "confusion", "curiosity",
                   "surprise", "caring"},
}

DRAMATIC_PAIRS = {
    ("positive", "negative"), ("negative", "positive"),
    ("positive", "aggressive"), ("aggressive", "positive"),
    ("negative", "aggressive"), ("aggressive", "negative"),
    ("positive", "anxious"),   ("anxious", "positive"),
}

def get_pole(emotion):
    for pole, emotions in EMOTION_POLES.items():
        if emotion in emotions:
            return pole
    return "neutral"

def get_transition_pause(prev_emotion, next_emotion):
    """Returns extra pre_break_ms when emotions shift poles dramatically."""
    if prev_emotion is None or prev_emotion == next_emotion:
        return 0
    prev_pole = get_pole(prev_emotion)
    next_pole = get_pole(next_emotion)
    if prev_pole == next_pole:
        return 0
    if (prev_pole, next_pole) in DRAMATIC_PAIRS:
        return 750   # 0.75s — dramatic shift
    return 500       # 0.5s — softer shift

# ---------------------------------------------------------------------------
# Intensity amplifier groups
# Applied AFTER base values — keeps base readable, extremes still hit hard
# ---------------------------------------------------------------------------

INTENSITY_GROUP = {
    "high":   {"anger", "excitement", "grief", "fear", "joy", "surprise"},
    "low":    {"neutral", "approval", "caring", "relief", "remorse"},
}

INTENSITY_AMPLIFIER = {
    "high":   1.20,   # reduced from 1.30 — prevents runaway rate
    "medium": 1.00,
    "low":    0.78,
}

def get_intensity_group(emotion):
    if emotion in INTENSITY_GROUP["high"]:   return "high"
    if emotion in INTENSITY_GROUP["low"]:    return "low"
    return "medium"

# ---------------------------------------------------------------------------
# Voice profiles — all 28 go_emotions
#
# Design principles:
#   - Base rates capped at ±30% — amplifier brings extremes to ~±36%
#   - Pitch and volume are the PRIMARY differentiators at emotional poles
#   - Clear separation between similar emotions (joy vs excitement, etc.)
#   - pre_break_ms: hesitation BEFORE clause (fear, grief, confusion)
#   - break_ms: natural pause AFTER clause
# ---------------------------------------------------------------------------

VOICE_PROFILES = {

    # ── Extreme positive (loud, fast, high) ─────────────────────────────────
    "excitement": {"rate": +30, "pitch": +15, "volume": +20,
                   "emphasis": "strong",   "pre_break_ms":   0, "break_ms":  80},
    "joy":        {"rate": +24, "pitch": +11, "volume": +16,
                   "emphasis": "strong",   "pre_break_ms":   0, "break_ms": 120},
    "surprise":   {"rate": +28, "pitch": +14, "volume": +15,
                   "emphasis": "strong",   "pre_break_ms":   0, "break_ms": 130},
    "amusement":  {"rate": +20, "pitch": +9,  "volume": +13,
                   "emphasis": "moderate", "pre_break_ms":   0, "break_ms": 140},

    # ── Warm positive (moderate pace, warm pitch) ────────────────────────────
    "love":       {"rate":  +8, "pitch":  +7, "volume": +10,
                   "emphasis": "moderate", "pre_break_ms":  80, "break_ms": 240},
    "gratitude":  {"rate":  +5, "pitch":  +5, "volume":  +7,
                   "emphasis": "moderate", "pre_break_ms":  60, "break_ms": 260},
    "admiration": {"rate":  +7, "pitch":  +5, "volume":  +8,
                   "emphasis": "moderate", "pre_break_ms":  50, "break_ms": 220},
    "pride":      {"rate": +11, "pitch":  +7, "volume": +12,
                   "emphasis": "moderate", "pre_break_ms":  40, "break_ms": 190},
    "optimism":   {"rate": +12, "pitch":  +6, "volume": +10,
                   "emphasis": "moderate", "pre_break_ms":  30, "break_ms": 190},
    "relief":     {"rate":  -7, "pitch":  +3, "volume":  -5,
                   "emphasis": "reduced",  "pre_break_ms": 160, "break_ms": 340},
    "approval":   {"rate":  +5, "pitch":  +3, "volume":  +5,
                   "emphasis": "moderate", "pre_break_ms":  40, "break_ms": 210},
    "desire":     {"rate":  +7, "pitch":  +5, "volume":  +6,
                   "emphasis": "moderate", "pre_break_ms":  60, "break_ms": 220},

    # ── Curious / inquisitive ───────────────────────────────────────────────
    "curiosity":  {"rate": +10, "pitch":  +7, "volume":  +5,
                   "emphasis": "moderate", "pre_break_ms":  40, "break_ms": 190},
    "confusion":  {"rate": -12, "pitch":  +5, "volume":  -6,
                   "emphasis": "reduced",  "pre_break_ms": 220, "break_ms": 400},
    "realization":{"rate": -16, "pitch":  +4, "volume":  -4,
                   "emphasis": "moderate", "pre_break_ms": 320, "break_ms": 440},

    # ── Soft / warm ─────────────────────────────────────────────────────────
    "caring":     {"rate": -10, "pitch":  +3, "volume":  -8,
                   "emphasis": "reduced",  "pre_break_ms": 100, "break_ms": 300},

    # ── Extreme negative (slow, very low pitch, quiet) ──────────────────────
    "grief":          {"rate": -30, "pitch": -15, "volume": -25,
                       "emphasis": "reduced",  "pre_break_ms": 380, "break_ms": 720},
    "sadness":        {"rate": -24, "pitch": -10, "volume": -19,
                       "emphasis": "reduced",  "pre_break_ms": 220, "break_ms": 520},
    "remorse":        {"rate": -20, "pitch":  -8, "volume": -16,
                       "emphasis": "reduced",  "pre_break_ms": 260, "break_ms": 480},
    "disappointment": {"rate": -18, "pitch":  -7, "volume": -13,
                       "emphasis": "reduced",  "pre_break_ms": 190, "break_ms": 440},
    "embarrassment":  {"rate": -14, "pitch":  -5, "volume": -11,
                       "emphasis": "reduced",  "pre_break_ms": 160, "break_ms": 360},
    "disgust":        {"rate":  -7, "pitch":  -5, "volume":  +8,
                       "emphasis": "moderate", "pre_break_ms":  90, "break_ms": 240},

    # ── Aggressive (fast, sharp, loud) ──────────────────────────────────────
    "anger":       {"rate": +30, "pitch": +10, "volume": +20,
                    "emphasis": "strong",   "pre_break_ms":   0, "break_ms":  85},
    "annoyance":   {"rate": +20, "pitch":  +6, "volume": +13,
                    "emphasis": "moderate", "pre_break_ms":  30, "break_ms": 150},
    "disapproval": {"rate": +13, "pitch":  +4, "volume":  +9,
                    "emphasis": "moderate", "pre_break_ms":  50, "break_ms": 190},

    # ── Anxious (fast but quieter, higher pitch) ─────────────────────────────
    "fear":        {"rate": +26, "pitch": +13, "volume":  -9,
                    "emphasis": "strong",   "pre_break_ms": 200, "break_ms": 170},
    "nervousness": {"rate": +18, "pitch":  +8, "volume":  -6,
                    "emphasis": "moderate", "pre_break_ms": 140, "break_ms": 210},

    # ── Baseline ────────────────────────────────────────────────────────────
    "neutral":     {"rate":   0, "pitch":   0, "volume":   0,
                    "emphasis": "none",     "pre_break_ms":   0, "break_ms": 200},
}

NEUTRAL_PARAMS = {
    "rate": 0, "pitch": 0, "volume": 0,
    "emphasis": "none", "pre_break_ms": 0, "break_ms": 200
}

# ---------------------------------------------------------------------------
# Personality baseline
# ---------------------------------------------------------------------------

PERSONALITY = {
    "expressiveness":  0.88,
    "volume_boost":    0.90,
    "pace_preference": 1.00,
}

# ---------------------------------------------------------------------------
# Conversational rules
# ---------------------------------------------------------------------------

CONVERSATIONAL_RULES = [
    {
        "name": "question",
        "trigger": lambda t: t.strip().endswith("?"),
        "modifier": {"rate": -6, "pitch": +5, "volume": 0,
                     "pre_break_ms": 0, "break_ms": +60},
    },
    {
        "name": "exclamation",
        "trigger": lambda t: t.strip().endswith("!"),
        "modifier": {"rate": +8, "pitch": +6, "volume": +8,
                     "pre_break_ms": 0, "break_ms": -30},
    },
    {
        "name": "trailing",
        "trigger": lambda t: t.strip().endswith("..."),
        "modifier": {"rate": -14, "pitch": -4, "volume": -10,
                     "pre_break_ms": +120, "break_ms": +220},
    },
    {
        "name": "emphasis",
        "trigger": lambda t: any(w.isupper() and len(w) > 1 for w in t.split()),
        "modifier": {"rate": +6, "pitch": +7, "volume": +12,
                     "pre_break_ms": 0, "break_ms": 0},
    },
    {
        "name": "soft_comma_pause",
        "trigger": lambda t: t.count(",") >= 2,
        "modifier": {"rate": -6, "pitch": 0, "volume": -3,
                     "pre_break_ms": +40, "break_ms": +70},
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

    total   = sum(e["score"] for e in candidates)
    blended = {"rate": 0.0, "pitch": 0.0, "volume": 0.0,
               "pre_break_ms": 0.0, "break_ms": 0.0}

    for e in candidates:
        weight  = e["score"] / total
        profile = VOICE_PROFILES.get(e["label"], NEUTRAL_PARAMS)
        for param in ("rate", "pitch", "volume", "pre_break_ms", "break_ms"):
            blended[param] += profile[param] * weight

    dominant       = candidates[0]
    dominant_label = dominant["label"]
    blended["emphasis"] = VOICE_PROFILES.get(dominant_label, NEUTRAL_PARAMS)["emphasis"]

    # Apply intensity amplifier on numeric params only
    amp = INTENSITY_AMPLIFIER[get_intensity_group(dominant_label)]
    for param in ("rate", "pitch", "volume"):
        blended[param] *= amp

    return blended, dominant_label, dominant["score"]

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
# Layer 4 — Dynamic Transition Smoothing
# ---------------------------------------------------------------------------

def smooth_transition(prev_params, next_params, alpha=0.25):
    """
    Dynamic alpha — extreme emotion shifts get less bleed-in
    so they hit with full impact instead of being softened.
    """
    if prev_params is None:
        return dict(next_params)

    rate_delta = abs(next_params["rate"] - prev_params.get("rate", 0))
    if rate_delta > 25:
        alpha = 0.08   # extreme shift — barely any carry-over
    elif rate_delta > 15:
        alpha = 0.16   # moderate shift
    # else keep default 0.25

    smoothed = dict(next_params)
    for param in ("rate", "pitch", "volume", "pre_break_ms", "break_ms"):
        smoothed[param] = (1 - alpha) * next_params[param] + alpha * prev_params.get(param, 0)
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
# Sentence position modifier
# ---------------------------------------------------------------------------

def apply_position_modifier(params, position, total_chunks, emotion_label):
    p = dict(params)
    if total_chunks == 1:
        return p

    heavy  = {"sadness", "grief", "remorse", "disappointment", "fear", "embarrassment"}
    punchy = {"anger", "excitement", "joy", "surprise", "annoyance"}

    if position == 0:
        # First clause — slow down, set the stage
        p["rate"]         = p["rate"] * 0.86
        p["pre_break_ms"] = p["pre_break_ms"] + 60

    elif position == total_chunks - 1:
        # Last clause — trail off or land hard
        if emotion_label in heavy:
            p["rate"]     = p["rate"]   * 0.84
            p["volume"]   = p["volume"] - 5
            p["break_ms"] = p["break_ms"] + 240
        elif emotion_label in punchy:
            p["rate"]     = p["rate"]   * 1.06
            p["volume"]   = p["volume"] + 4
            p["break_ms"] = max(80, p["break_ms"] - 40)

    return p

# ---------------------------------------------------------------------------
# Layer 2 — Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(text):
    raw    = re.split(r'(?<=[.!?,;])\s+', text.strip())
    chunks = [c.strip() for c in raw if c.strip()]
    return chunks if chunks else [text.strip()]

# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def get_chunk_params(emotion_outputs, chunk_text_str, prev_params=None,
                     prev_emotion=None, position=0, total_chunks=1):
    # Layer 1 — blend
    blended, dominant_emotion, dominant_score = blend_voice_params(emotion_outputs)
    # Layer 3 — conversational rules
    blended = apply_conversational_rules(blended, chunk_text_str)
    # Layer 4 — dynamic smoothing
    blended = smooth_transition(prev_params, blended)
    # Layer 5 — personality
    blended = apply_personality(blended, dominant_emotion)
    # Position modifier
    blended = apply_position_modifier(blended, position, total_chunks, dominant_emotion)
    # Transition pause on pole change
    transition_ms = get_transition_pause(prev_emotion, dominant_emotion)
    blended["pre_break_ms"] = blended["pre_break_ms"] + transition_ms

    blended["dominant_emotion"] = dominant_emotion
    blended["dominant_score"]   = round(dominant_score, 4)
    return blended
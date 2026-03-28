"""
voice_map.py — The brain of the Empathy Engine.

Pipeline:
  0. Pre-pass            — global speed averaging across all chunks
  1. Emotion Blending    — weighted blend of top-3 emotions
  2. Prosody Chunking    — per-clause emotion detection
  3. Conversational Rules— punctuation / caps / special char overrides
  3b. Micro-emotion words— sigh, huh, taunt, cry detection
  4. Transition Smoothing— dynamic alpha + rate bridging on sudden shifts
  4b. Dialogue detection — faster tone shift on quoted speech
  5. Personality Baseline— consistent voice character
  6. Position modifier   — first/last clause handling
  7. Pole transition     — 0.5–0.8s pause on dramatic emotion changes
"""

import re

# ---------------------------------------------------------------------------
# Emotion poles
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
    if prev_emotion is None or prev_emotion == next_emotion:
        return 0
    prev_pole = get_pole(prev_emotion)
    next_pole = get_pole(next_emotion)
    if prev_pole == next_pole:
        return 0
    if (prev_pole, next_pole) in DRAMATIC_PAIRS:
        return 750
    return 500

# ---------------------------------------------------------------------------
# Intensity amplifier groups
# ---------------------------------------------------------------------------

INTENSITY_GROUP = {
    "high": {"anger", "excitement", "grief", "fear", "joy", "surprise"},
    "low":  {"neutral", "approval", "caring", "relief", "remorse"},
}

INTENSITY_AMPLIFIER = {"high": 1.20, "medium": 1.00, "low": 0.78}

def get_intensity_group(emotion):
    if emotion in INTENSITY_GROUP["high"]: return "high"
    if emotion in INTENSITY_GROUP["low"]:  return "low"
    return "medium"

# ---------------------------------------------------------------------------
# Micro-emotion word patterns (Rule 5)
# Detected per chunk — override or boost params
# ---------------------------------------------------------------------------

MICRO_EMOTIONS = [
    {
        "name": "sigh",
        "pattern": re.compile(r'\b(sigh|ugh|phew|haah|hah)\b', re.IGNORECASE),
        "modifier": {"rate": -18, "pitch": -6, "volume": -12, "pre_break_ms": +150, "break_ms": +200},
    },
    {
        "name": "cry",
        "pattern": re.compile(r'\b(sob|crying|sniffling|weeping|teary)\b', re.IGNORECASE),
        "modifier": {"rate": -20, "pitch": -8, "volume": -15, "pre_break_ms": +200, "break_ms": +300},
    },
    {
        "name": "huh",
        # Huh? Huhhh? — stretched vowel = slower rate, raised pitch
        "pattern": re.compile(r'\bhu+h+\??\b', re.IGNORECASE),
        "modifier": {"rate": -22, "pitch": +8,  "volume": +5,  "pre_break_ms": +80,  "break_ms": +180},
    },
    {
        "name": "taunt",
        "pattern": re.compile(r'\b(oh really|sure sure|right right|obviously|clearly|wow okay|as if)\b', re.IGNORECASE),
        "modifier": {"rate": +10, "pitch": +9,  "volume": +8,  "pre_break_ms": +40,  "break_ms": +100},
    },
    {
        "name": "emphasis_word",
        # Words in *asterisks* or _underscores_ = emphasis
        "pattern": re.compile(r'[*_][^*_]+[*_]'),
        "modifier": {"rate": -8,  "pitch": +6,  "volume": +10, "pre_break_ms": +30,  "break_ms": +60},
    },
    {
        "name": "hmm",
        "pattern": re.compile(r'\bh+m+\b', re.IGNORECASE),
        "modifier": {"rate": -15, "pitch": +3,  "volume": -8,  "pre_break_ms": +100, "break_ms": +160},
    },
]

def apply_micro_emotions(params, text):
    """Detect micro-emotion words and boost their params."""
    p = dict(params)
    for me in MICRO_EMOTIONS:
        if me["pattern"].search(text):
            for param, delta in me["modifier"].items():
                if param in p and isinstance(p[param], (int, float)):
                    p[param] = p[param] + delta
    return p

# ---------------------------------------------------------------------------
# Dialogue detection (Rule 4b)
# ---------------------------------------------------------------------------

DIALOGUE_MARKERS = re.compile(
    r'(he said|she said|they said|i said|said he|said she|'
    r'he asked|she asked|they asked|he replied|she replied|'
    r'he whispered|she whispered|he shouted|she shouted)',
    re.IGNORECASE
)

def is_dialogue(text):
    """Returns True if chunk appears to be quoted/reported speech."""
    has_quotes  = bool(re.search(r'["\u201c\u201d]', text))
    has_markers = bool(DIALOGUE_MARKERS.search(text))
    return has_quotes or has_markers

# ---------------------------------------------------------------------------
# Voice profiles — all 28 go_emotions
# ---------------------------------------------------------------------------

VOICE_PROFILES = {
    # ── Extreme positive ────────────────────────────────────────────────────
    "excitement": {"rate": +30, "pitch": +15, "volume": +20, "emphasis": "strong",   "pre_break_ms":   0, "break_ms":  80},
    "joy":        {"rate": +24, "pitch": +11, "volume": +16, "emphasis": "strong",   "pre_break_ms":   0, "break_ms": 120},
    "surprise":   {"rate": +28, "pitch": +14, "volume": +15, "emphasis": "strong",   "pre_break_ms":   0, "break_ms": 130},
    "amusement":  {"rate": +20, "pitch":  +9, "volume": +13, "emphasis": "moderate", "pre_break_ms":   0, "break_ms": 140},

    # ── Warm positive ───────────────────────────────────────────────────────
    "love":       {"rate":  +8, "pitch":  +7, "volume": +10, "emphasis": "moderate", "pre_break_ms":  80, "break_ms": 240},
    "gratitude":  {"rate":  +5, "pitch":  +5, "volume":  +7, "emphasis": "moderate", "pre_break_ms":  60, "break_ms": 260},
    "admiration": {"rate":  +7, "pitch":  +5, "volume":  +8, "emphasis": "moderate", "pre_break_ms":  50, "break_ms": 220},
    "pride":      {"rate": +11, "pitch":  +7, "volume": +12, "emphasis": "moderate", "pre_break_ms":  40, "break_ms": 190},
    "optimism":   {"rate": +12, "pitch":  +6, "volume": +10, "emphasis": "moderate", "pre_break_ms":  30, "break_ms": 190},
    "relief":     {"rate":  -7, "pitch":  +3, "volume":  -5, "emphasis": "reduced",  "pre_break_ms": 160, "break_ms": 340},
    "approval":   {"rate":  +5, "pitch":  +3, "volume":  +5, "emphasis": "moderate", "pre_break_ms":  40, "break_ms": 210},
    "desire":     {"rate":  +7, "pitch":  +5, "volume":  +6, "emphasis": "moderate", "pre_break_ms":  60, "break_ms": 220},

    # ── Curious / inquisitive ───────────────────────────────────────────────
    "curiosity":   {"rate": +10, "pitch":  +7, "volume":  +5, "emphasis": "moderate", "pre_break_ms":  40, "break_ms": 190},
    "confusion":   {"rate": -12, "pitch":  +5, "volume":  -6, "emphasis": "reduced",  "pre_break_ms": 220, "break_ms": 400},
    "realization": {"rate": -16, "pitch":  +4, "volume":  -4, "emphasis": "moderate", "pre_break_ms": 320, "break_ms": 440},
    "surprise":    {"rate": +28, "pitch": +14, "volume": +15, "emphasis": "strong",   "pre_break_ms":   0, "break_ms": 130},

    # ── Soft / warm ─────────────────────────────────────────────────────────
    "caring":      {"rate": -10, "pitch":  +3, "volume":  -8, "emphasis": "reduced",  "pre_break_ms": 100, "break_ms": 300},

    # ── Extreme negative ────────────────────────────────────────────────────
    "grief":           {"rate": -30, "pitch": -15, "volume": -25, "emphasis": "reduced",  "pre_break_ms": 380, "break_ms": 720},
    "sadness":         {"rate": -24, "pitch": -10, "volume": -19, "emphasis": "reduced",  "pre_break_ms": 220, "break_ms": 520},
    "remorse":         {"rate": -20, "pitch":  -8, "volume": -16, "emphasis": "reduced",  "pre_break_ms": 260, "break_ms": 480},
    "disappointment":  {"rate": -18, "pitch":  -7, "volume": -13, "emphasis": "reduced",  "pre_break_ms": 190, "break_ms": 440},
    "embarrassment":   {"rate": -14, "pitch":  -5, "volume": -11, "emphasis": "reduced",  "pre_break_ms": 160, "break_ms": 360},
    "disgust":         {"rate":  -7, "pitch":  -5, "volume":  +8, "emphasis": "moderate", "pre_break_ms":  90, "break_ms": 240},

    # ── Aggressive ──────────────────────────────────────────────────────────
    "anger":       {"rate": +30, "pitch": +10, "volume": +20, "emphasis": "strong",   "pre_break_ms":   0, "break_ms":  85},
    "annoyance":   {"rate": +20, "pitch":  +6, "volume": +13, "emphasis": "moderate", "pre_break_ms":  30, "break_ms": 150},
    "disapproval": {"rate": +13, "pitch":  +4, "volume":  +9, "emphasis": "moderate", "pre_break_ms":  50, "break_ms": 190},

    # ── Anxious ─────────────────────────────────────────────────────────────
    "fear":        {"rate": +26, "pitch": +13, "volume":  -9, "emphasis": "strong",   "pre_break_ms": 200, "break_ms": 170},
    "nervousness": {"rate": +18, "pitch":  +8, "volume":  -6, "emphasis": "moderate", "pre_break_ms": 140, "break_ms": 210},

    # ── Baseline ────────────────────────────────────────────────────────────
    "neutral":     {"rate":   0, "pitch":   0, "volume":   0, "emphasis": "none",     "pre_break_ms":   0, "break_ms": 200},
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
# Conversational rules (Rule 3 — improved)
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
        "modifier": {"rate": +10, "pitch": +7, "volume": +10,
                     "pre_break_ms": 0, "break_ms": -30},
    },
    {
        "name": "double_exclamation",
        "trigger": lambda t: t.strip().endswith("!!") or t.strip().endswith("!!!"),
        "modifier": {"rate": +16, "pitch": +10, "volume": +14,
                     "pre_break_ms": 0, "break_ms": -40},
    },
    {
        "name": "trailing",
        "trigger": lambda t: t.strip().endswith("..."),
        "modifier": {"rate": -16, "pitch": -5, "volume": -12,
                     "pre_break_ms": +140, "break_ms": +240},
    },
    {
        "name": "em_dash",
        # Em dash or double dash = abrupt cut or dramatic pause
        "trigger": lambda t: "—" in t or "--" in t,
        "modifier": {"rate": -8, "pitch": +2, "volume": -4,
                     "pre_break_ms": +60, "break_ms": +120},
    },
    {
        "name": "semicolon",
        # Semicolon = soft period, slightly longer pause
        "trigger": lambda t: ";" in t,
        "modifier": {"rate": -4, "pitch": 0, "volume": -2,
                     "pre_break_ms": +30, "break_ms": +80},
    },
    {
        "name": "emphasis_caps",
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
    {
        "name": "quoted_speech",
        # Text in quotes = slight pitch shift for reported speech
        "trigger": lambda t: bool(re.search(r'["\u201c\u201d]', t)),
        "modifier": {"rate": +4, "pitch": +4, "volume": +3,
                     "pre_break_ms": +20, "break_ms": +40},
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

def smooth_transition(prev_params, next_params, alpha=0.25, dialogue=False):
    """
    Dynamic alpha based on rate delta.
    Dialogue detected → alpha near 0 for instant character shift.
    """
    if prev_params is None:
        return dict(next_params)

    if dialogue:
        alpha = 0.04  # almost instant shift for new speaker

    else:
        rate_delta = abs(next_params["rate"] - prev_params.get("rate", 0))
        if rate_delta > 25:
            alpha = 0.08
        elif rate_delta > 15:
            alpha = 0.16

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
        p["rate"]         = p["rate"] * 0.86
        p["pre_break_ms"] = p["pre_break_ms"] + 60

    elif position == total_chunks - 1:
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
# Pre-pass — Global Speed Averaging (Rule 1)
# ---------------------------------------------------------------------------

def compute_global_rate_anchor(all_raw_rates, max_deviation=18):
    """
    Compute mean rate across all chunks.
    Each chunk's rate is then constrained to mean ± max_deviation.
    This prevents jarring speed jumps while preserving emotional direction.
    """
    if not all_raw_rates:
        return 0
    return sum(all_raw_rates) / len(all_raw_rates)

def apply_rate_anchor(rate, anchor, max_deviation=18):
    """Clamp rate to anchor ± max_deviation."""
    lo = anchor - max_deviation
    hi = anchor + max_deviation
    return max(lo, min(hi, rate))

# ---------------------------------------------------------------------------
# Master pipeline — get params for a single chunk
# ---------------------------------------------------------------------------

def get_chunk_params(emotion_outputs, chunk_text_str, prev_params=None,
                     prev_emotion=None, position=0, total_chunks=1,
                     rate_anchor=None):
    # Layer 1
    blended, dominant_emotion, dominant_score = blend_voice_params(emotion_outputs)
    # Layer 3
    blended = apply_conversational_rules(blended, chunk_text_str)
    # Rule 3b — micro emotions
    blended = apply_micro_emotions(blended, chunk_text_str)
    # Layer 4 — dialogue-aware smoothing
    dialogue = is_dialogue(chunk_text_str)
    blended  = smooth_transition(prev_params, blended, dialogue=dialogue)
    # Layer 5
    blended  = apply_personality(blended, dominant_emotion)
    # Position
    blended  = apply_position_modifier(blended, position, total_chunks, dominant_emotion)
    # Rule 1 — apply rate anchor if provided
    if rate_anchor is not None:
        blended["rate"] = apply_rate_anchor(blended["rate"], rate_anchor)
    # Rule 2 — pole transition pause
    transition_ms = get_transition_pause(prev_emotion, dominant_emotion)
    blended["pre_break_ms"] = blended["pre_break_ms"] + transition_ms

    blended["dominant_emotion"] = dominant_emotion
    blended["dominant_score"]   = round(dominant_score, 4)
    return blended
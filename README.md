# 🎙 Empathy Engine

> Dynamic vocal synthesis that maps emotion to voice — not as a switch, but as a human would.

## What it does

The Empathy Engine takes text, detects emotion **per clause**, and synthesizes speech where every sentence breathes differently. It uses a 5-layer pipeline to produce audio that feels genuinely human rather than robotically consistent.

---

## Architecture: The 5-Layer Pipeline

```
Input text
    │
    ▼
[Layer 2] chunk_text()
    Split into prosodic clauses at punctuation boundaries
    │
    ▼ (per chunk)
[Layer 1] blend_voice_params()
    Weighted blend of top-3 emotions (not just top-1)
    Uses: SamLowe/roberta-base-go_emotions-onnx (INT8 quantized, 28 emotions)
    │
    ▼
[Layer 3] apply_conversational_rules()
    Punctuation overrides: ?, !, ..., ALL CAPS, dense commas
    │
    ▼
[Layer 4] smooth_transition()
    Blend 25% of previous chunk state into current
    (emotion doesn't teleport between sentences)
    │
    ▼
[Layer 5] apply_personality()
    Global voice character: expressiveness, pace, warmth
    │
    ▼
ElevenLabs TTS  →  stitch chunks  →  final .mp3
```

### Why this produces human-sounding output

| Human trait | How we model it |
|---|---|
| Emotions blend, not switch | Weighted top-3 blend (Layer 1) |
| Each clause has its own feel | Per-clause detection (Layer 2) |
| Punctuation changes delivery | Conversational rules (Layer 3) |
| Emotion carries over sentences | Smoothing with alpha=0.25 (Layer 4) |
| Consistent personality | Global baseline modifiers (Layer 5) |

---

## ElevenLabs Voice Parameters

| Param | Range | Role |
|---|---|---|
| `stability` | 0–1 | Lower = more expressive. Angry/excited → 0.18–0.28. Sad/calm → 0.70–0.80 |
| `style` | 0–1 | Dramatic delivery exaggeration. Anger → 0.94. Neutral → 0.30 |
| `speed` | 0.7–1.2 | Pace. Anger/excitement → 1.15–1.18. Grief → 0.76 |
| `similarity_boost` | fixed 0.75 | Voice identity consistency |

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd empathy-engine
pip install -r requirements.txt
```

Also install `ffmpeg` (required by pydub for mp3 handling):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 2. Set environment variables

```bash
export ELEVENLABS_API_KEY="your_api_key_here"
export ELEVENLABS_VOICE_ID="21m00Tcm4TlvDq8ikWAM"   # Rachel (default)
```

Get your API key from: https://elevenlabs.io  
Find voice IDs at: https://api.elevenlabs.io/v1/voices

### 3. Run

```bash
python app.py
```

Open http://localhost:5000

---

## The Model

**SamLowe/roberta-base-go_emotions-onnx** (INT8 quantized)

- 28 emotions from the GoEmotions dataset
- Multi-label: multiple emotions can be active at once
- 125MB, ~5x faster than PyTorch equivalent
- Threshold: 0.15 (lower than default 0.5 to catch secondary emotions for blending)

All 28 labels:
`admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral`

---

## Design choices

**Why blend top-3 instead of top-1?**  
Real speech rarely expresses a single pure emotion. "I can't believe you did that" might score 0.6 disappointment + 0.3 sadness + 0.2 anger — and the voice should reflect all three.

**Why per-clause chunking?**  
"I thought we'd win. We didn't." — first clause is hopeful, second is deflated. A single-text approach would average them into something neither feels.

**Why smoothing?**  
Hard jumps between emotional states sound robotic. The 25% carry-over from the previous chunk simulates how human vocal tone trails from one sentence into the next.

**Why the personality layer?**  
Without a baseline, the voice feels like it has no consistent identity. The personality layer ensures the same "character" speaks throughout, regardless of emotional swings.

---

## Project structure

```
empathy-engine/
├── app.py          # Fast + API endpoints
├── engine.py       # Pipeline orchestrator
├── emotion.py      # HF ONNX emotion detection
├── voice_map.py    # All 5 layers of voice param logic
├── tts.py          # ElevenLabs synthesis + audio stitching
├── templates/
│   └── index.html  # Web UI with per-clause breakdown
├── outputs/        # Generated mp3 files
└── requirements.txt
```

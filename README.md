# 🎙 Empathy Engine

> Dynamic vocal synthesis that maps emotion to voice — not as a switch, but as a human would.

## Folder Structure

```
empathy-engine/
├── app.py              # FastAPI server
├── engine.py           # Pipeline orchestrator
├── emotion.py          # ONNX emotion detection
├── voice_map.py        # 5-layer voice param logic + SSML builder
├── tts.py              # edge-tts synthesis
├── requirements.txt
├── templates/
│   └── index.html      # Web UI
└── outputs/            # Auto-created — generated mp3s land here
```

## Setup

```bash
pip install -r requirements.txt
```

## Environment Variables

```
EDGE_TTS_VOICE=en-US-JennyNeural   # optional, this is the default
```

No API keys required. edge-tts is free.

## Run

```bash
python app.py
```

Open http://localhost:8000

## The Pipeline

```
Input text
    │
    ▼
chunk_text()               # split into clauses at punctuation
    │
    ▼ (per chunk)
classify()                 # 28-emotion ONNX model (top-3)
    │
    ▼
blend_voice_params()       # weighted blend of top-3 emotions
    │
    ▼
apply_conversational_rules()  # ?, !, ..., CAPS overrides
    │
    ▼
smooth_transition()        # 25% carry-over from previous chunk
    │
    ▼
apply_personality()        # global expressiveness/pace/volume
    │
    ▼
build_ssml()               # wrap in <prosody> + <emphasis> + <break>
    │
    ▼
edge-tts                   # single synthesis call → mp3
```

## Available Voices (EDGE_TTS_VOICE)

| Voice | Character |
|---|---|
| en-US-JennyNeural | Warm, expressive — default |
| en-US-AriaNeural  | Slightly more dramatic |
| en-US-GuyNeural   | Male alternative |
| en-GB-SoniaNeural | British female |
| en-GB-RyanNeural  | British male |
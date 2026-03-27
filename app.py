"""
app.py — FastAPI server for the Empathy Engine.

Endpoints:
  GET  /              → Web UI
  POST /synthesize    → JSON: {text} → {output_path, chunks, dominant, processing_ms}
  GET  /audio/<file>  → Serve generated mp3
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from engine import process

app = FastAPI(title="Empathy Engine", version="1.0.0")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@app.get("/", response_class=HTMLResponse)
async def index():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r") as f:
        return f.read()


@app.post("/synthesize")
async def synthesize_route(body: SynthesizeRequest):
    try:
        result = process(body.text.strip())
        filename = os.path.basename(result["output_path"])
        result["audio_url"] = f"/audio/{filename}"
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=traceback.format_exc())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

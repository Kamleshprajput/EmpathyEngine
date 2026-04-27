"""
app.py — FastAPI server for the Empathy Engine.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
from engine import process
from emotion import load_classifier
from cache import get_cached, set_cached, cache_stats

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the model before serving any requests
    print("[app] Pre-loading emotion classifier...")
    load_classifier()
    print("[app] Classifier ready.")
    yield


app = FastAPI(title="Empathy Engine", version="1.0.0", lifespan=lifespan)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")

executor = ThreadPoolExecutor(max_workers=2)


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r") as f:
        return f.read()

@app.get("/cache/stats")                                 # ← NEW
async def cache_stats_route():
    return cache_stats()
@app.post("/synthesize")
async def synthesize_route(body: SynthesizeRequest):
    text = body.text.strip()
 
    # ── 1. Cache lookup ───────────────────────────────────────────────────────
    cached = get_cached(text)
    if cached is not None:
        cached["cached"] = True
        return JSONResponse(content=cached)
 
    # ── 2. Full synthesis pipeline ────────────────────────────────────────────
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, process, text),
            timeout=120.0,
        )
        filename = os.path.basename(result["output_path"])
        result["audio_url"] = f"/audio/{filename}"
        result["cached"] = False
 
        # ── 3. Store in cache ─────────────────────────────────────────────────
        set_cached(text, result)
 
        return JSONResponse(content=result)
 
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Synthesis timed out. Try shorter text or retry.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
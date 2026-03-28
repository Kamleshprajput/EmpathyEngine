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


@app.post("/synthesize")
async def synthesize_route(body: SynthesizeRequest):
    try:
        loop   = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, process, body.text.strip()),
            timeout=120.0   # 2 min timeout — enough for model load + synthesis
        )
        filename = os.path.basename(result["output_path"])
        result["audio_url"] = f"/audio/{filename}"
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
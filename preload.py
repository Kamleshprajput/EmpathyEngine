"""
preload.py — Run during build to cache the ONNX model into the image.
This avoids downloading it on first request (which causes cold-start timeouts).
"""
from emotion import load_classifier

print("[preload] Downloading and caching ONNX model...")
load_classifier()
print("[preload] Done.")

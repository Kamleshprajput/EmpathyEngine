"""
preload.py — Run during build to cache the ONNX model into the image.
This avoids downloading it on first request (which causes cold-start timeouts).
"""
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

MODEL_ID = "SamLowe/roberta-base-go_emotions-onnx"

print("[preload] Downloading and caching ONNX model...")
ORTModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    subfolder="onnx",
    file_name="model_quantized.onnx"
)
AutoTokenizer.from_pretrained(MODEL_ID)
print("[preload] Model cached successfully.")

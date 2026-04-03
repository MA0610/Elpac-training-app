"""
Quantize wav2vec2_phoneme.onnx to int8 for Android.

Run after export_model.py:
    pip install onnx onnxruntime
    python quantize_model.py

Input:  wav2vec2_phoneme.onnx   (~1.2 GB float32)
Output: wav2vec2_phoneme.onnx   (~300 MB int8, overwrites original)
"""

import os
from onnxruntime.quantization import quantize_dynamic, QuantType

SRC = "wav2vec2_phoneme.onnx"
DST = "wav2vec2_phoneme_q.onnx"

print("Quantizing transformer weights to int8 (skipping CNN conv layers)...")
print("This may take a few minutes...")

# Quantize only MatMul + Gather — the transformer attention/FFN layers that
# make up ~90% of model size. Skipping Conv avoids the initializer error.
quantize_dynamic(
    SRC, DST,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gather"],
)

src_mb = os.path.getsize(SRC) / 1_048_576
dst_mb = os.path.getsize(DST) / 1_048_576
print(f"Original: {src_mb:.0f} MB  →  Quantized: {dst_mb:.0f} MB")

# Overwrite original so the rest of the pipeline uses this file
os.replace(DST, SRC)
print(f"Saved as {SRC}")
print()
print("Next steps:")
print("  ~/Library/Android/sdk/platform-tools/adb push wav2vec2_phoneme.onnx /data/local/tmp/wav2vec2_phoneme.onnx")
print("  ~/Library/Android/sdk/platform-tools/adb shell run-as com.example.vosk_elpac cp /data/local/tmp/wav2vec2_phoneme.onnx /data/data/com.example.vosk_elpac/files/wav2vec2_phoneme.onnx")

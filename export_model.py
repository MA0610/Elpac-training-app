"""
Export the local 'age aware base +' WavLM-CTC checkpoint to ONNX for Android.

Requirements:
    pip install transformers torch onnx

Usage:
    python export_model.py

The checkpoint directory './age aware base +/' must exist in the repo root.
This is the WavLM-base model fine-tuned with a CTC head (~94 M parameters).

Outputs:
    wavlm_phoneme.onnx   (~360 MB float32, single self-contained file)
    wavlm_vocab.json     (token_id → IPA symbol, copy to app/src/main/assets/)

After export:
    1. Copy wavlm_vocab.json to app/src/main/assets/wavlm_vocab.json
    2. Upload wavlm_phoneme.onnx to GitHub Releases (tag: v2.0)
    3. Update WAVLM_MODEL_URL and WAVLM_MODEL_SHA256 in app/build.gradle
"""

import json
import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

CHECKPOINT_DIR = "./age aware base +/"
OUT_ONNX  = "wavlm_phoneme.onnx"
OUT_VOCAB = "wavlm_vocab.json"

# ── 1. Load model from local checkpoint ───────────────────────────────────────
print(f"Loading checkpoint from {CHECKPOINT_DIR!r}...")
if not os.path.isdir(CHECKPOINT_DIR):
    raise FileNotFoundError(
        f"Checkpoint directory not found: {CHECKPOINT_DIR!r}\n"
        "Ensure the 'age aware base +' directory is present in the repo root."
    )

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(CHECKPOINT_DIR)
model     = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT_DIR, torch_dtype=torch.float32)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params / 1e6:.0f} M  (WavLM-base + CTC head, expect ~94 M)")

# ── 2. Wrapper that returns only logits (required for ONNX traceability) ──────
class WavLMOnnxWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.m(input_values=input_values).logits

wrapper = WavLMOnnxWrapper(model)

# Dummy input: 1-second batch at 16 kHz
dummy_audio = torch.randn(1, 16000)

# Sanity-check: verify the model actually runs
with torch.no_grad():
    logits = wrapper(dummy_audio)
print(f"Sanity check — logits shape: {logits.shape}  (expect [1, ~49, vocab_size])")
if logits.shape[-1] < 10:
    raise RuntimeError("Logits vocab dimension is too small — model may not have loaded correctly.")

# ── 3. Export to ONNX ─────────────────────────────────────────────────────────
print(f"Exporting to {OUT_ONNX}  (this may take a few minutes)...")
with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (dummy_audio,),
        OUT_ONNX,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "sequence"},
            "logits":       {0: "batch", 1: "frames"},
        },
        opset_version=14,
        do_constant_folding=False,
    )

# ── 3b. Merge external data into a single self-contained ONNX file ────────────
data_file = OUT_ONNX + ".data"
if os.path.exists(data_file):
    print("Merging external data into single file...")
    import onnx
    merged_model = onnx.load(OUT_ONNX, load_external_data=True)
    onnx.save(merged_model, OUT_ONNX, save_as_external_data=False)
    os.remove(data_file)
    print("Merged — .data file removed.")

size_mb = os.path.getsize(OUT_ONNX) / 1_048_576
print(f"Saved: {OUT_ONNX}  ({size_mb:.0f} MB)")
if size_mb < 100:
    print("WARNING: file is unexpectedly small — model may not have exported correctly.")
else:
    print("Size looks correct.")

# ── 4. Export vocab ───────────────────────────────────────────────────────────
vocab = {int(v): k for k, v in tokenizer.get_vocab().items()}
with open(OUT_VOCAB, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2, sort_keys=True)
print(f"Saved: {OUT_VOCAB}  ({len(vocab)} tokens)")

print()
print("Next steps:")
print(f"  1. Copy {OUT_VOCAB} to app/src/main/assets/wavlm_vocab.json")
print(f"  2. Upload {OUT_ONNX} to GitHub Releases (tag: v2.0)")
print("  3. Update WAVLM_MODEL_URL and WAVLM_MODEL_SHA256 in app/build.gradle")
print("  4. ./gradlew assembleDebug && adb install ...")

"""
Export facebook/wav2vec2-lv-60-espeak-cv-ft to ONNX for Android.

Run once before uploading to GitHub Releases:
    pip install transformers torch
    python export_model.py

Outputs:
    wav2vec2_phoneme.onnx   (~380 MB float32)
    wav2vec2_vocab.json     (token_id → IPA symbol)

Then upload wav2vec2_phoneme.onnx to:
    https://github.com/MA0610/Vosk_Elpac/releases/new  (tag: v1.0)
"""

import json
import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
OUT_ONNX  = "wav2vec2_phoneme.onnx"
OUT_VOCAB = "wav2vec2_vocab.json"

# ── 1. Load model ─────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID}...")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_ID)
model     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params / 1e6:.0f} M  (expect ~300 M for wav2vec2-large)")

# ── 2. Wrapper that returns only logits (required for ONNX traceability) ──────
class Wav2Vec2OnnxWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.m(input_values=input_values, attention_mask=attention_mask).logits

wrapper = Wav2Vec2OnnxWrapper(model)

# Dummy inputs: 1-second batch
dummy_audio = torch.randn(1, 16000)
dummy_mask  = torch.ones(1, 16000, dtype=torch.long)

# Sanity-check: verify the model actually runs
with torch.no_grad():
    logits = wrapper(dummy_audio, dummy_mask)
print(f"Sanity check — logits shape: {logits.shape}  (expect [1, ~49, vocab_size])")
if logits.shape[-1] < 10:
    raise RuntimeError("Logits vocab dimension is too small — model may not have loaded correctly.")

# ── 3. Export to ONNX ─────────────────────────────────────────────────────────
print(f"Exporting to {OUT_ONNX}  (this may take a few minutes)...")
with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (dummy_audio, dummy_mask),
        OUT_ONNX,
        input_names=["input_values", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_values":   {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits":         {0: "batch", 1: "frames"},
        },
        opset_version=14,
        do_constant_folding=False,
    )

# ── 3b. Merge external data into a single self-contained ONNX file ────────────
# Large models split into .onnx + .onnx.data — merge so only one file needs uploading.
data_file = OUT_ONNX + ".data"
if os.path.exists(data_file):
    print("Merging external data into single file...")
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
    merged_model = onnx.load(OUT_ONNX, load_external_data=True)
    onnx.save(merged_model, OUT_ONNX,
              save_as_external_data=False)   # inline all tensors
    os.remove(data_file)
    print("Merged — .data file removed.")

size_mb = os.path.getsize(OUT_ONNX) / 1_048_576
print(f"Saved: {OUT_ONNX}  ({size_mb:.0f} MB)")
if size_mb < 100:
    print("WARNING: file is unexpectedly small — model may not have exported correctly.")
else:
    print("✅ Size looks correct.")

# ── 4. Export vocab ───────────────────────────────────────────────────────────
vocab = {int(v): k for k, v in tokenizer.get_vocab().items()}
with open(OUT_VOCAB, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2, sort_keys=True)
print(f"Saved: {OUT_VOCAB}  ({len(vocab)} tokens)")

print()
print("Next steps:")
print(f"  1. Upload {OUT_ONNX} to https://github.com/MA0610/Vosk_Elpac/releases/new (tag: v1.0)")
print(f"  2. Copy {OUT_VOCAB} to Vosk_Elpac/app/src/main/assets/wav2vec2_vocab.json")
print("  3. ./gradlew assembleDebug && adb install ...")

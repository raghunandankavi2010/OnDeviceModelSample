#!/usr/bin/env python3
"""
Converts VisionaryQuant/5_Crop_Disease_Detection (PyTorch EfficientNet-B3) to
TFLite for on-device inference in the Android app.

The HF repo only ships a raw `best_crop_disease_model.pt` — no transformers
config, no preprocessor, no pipeline_tag — so HF's Inference API can't auto-route
it. We download the weights, reconstruct EfficientNet-B3 with a 17-class head,
export to TFLite, and write the model + labels into app/src/main/assets/.

Usage (from project root):

    python -m venv .venv && source .venv/bin/activate
    pip install --upgrade pip
    pip install torch torchvision huggingface_hub ai-edge-torch numpy pillow
    # Fallback path also needs: onnx onnxruntime onnx-tf tensorflow

    python tools/convert_crop_disease_to_tflite.py

Outputs:
    app/src/main/assets/crop_disease_model.tflite
    app/src/main/assets/crop_disease_labels.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import json
import subprocess

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

REPO_ID = "VisionaryQuant/5_Crop_Disease_Detection"
WEIGHTS_FILENAME = "best_crop_disease_model.pt"
INPUT_SIZE = 300
NUM_CLASSES = 17

# Order follows the HF model card. If the parity check below disagrees with
# the model's actual training-time order, re-order this list (and verify with
# a few known images) before shipping.
CLASS_LABELS = [
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Northern_Leaf_Blight",
    "Corn___Healthy",
    "Potato___Early_Blight",
    "Potato___Late_Blight",
    "Potato___Healthy",
    "Rice___Brown_Spot",
    "Rice___Leaf_Blast",
    "Rice___Neck_Blast",
    "Rice___Healthy",
    "Wheat___Yellow_Rust",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Sugarcane___Red_Rot",
    "Sugarcane___Bacterial_Blight",
    "Sugarcane___Healthy",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "app" / "src" / "main" / "assets"


def build_model() -> nn.Module:
    # The checkpoint uses timm naming (conv_stem, bn1, blocks.X.Y.conv_dw/se/...,
    # conv_head, bn2) and wraps the classifier in nn.Sequential — matching keys
    # `classifier.0.weight` of shape (17, 1536). Build accordingly.
    import timm
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=17)
    in_features = model.classifier.in_features  # 1536
    model.classifier = nn.Sequential(nn.Linear(in_features, NUM_CLASSES))
    return model


def load_state_dict(model: nn.Module, weights_path: str) -> None:
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, nn.Module):
        state = state.state_dict()
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [warn] missing keys ({len(missing)}): first few = {missing[:5]}")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): first few = {unexpected[:5]}")
    if missing or unexpected:
        print("  [warn] If many keys are missing the architecture may not match. "
              "The model card says EfficientNet-B3 + Linear(1536, 17); if this script "
              "fails parity, try `timm.create_model('tf_efficientnet_b3', ...)` instead.")
    model.eval()


def export_with_ai_edge_torch(model: nn.Module, out_path: Path) -> bool:
    # The package was renamed `ai-edge-torch` → `litert-torch`; try the new
    # name first, then fall back to the deprecated alias.
    converter_mod = None
    for name in ("litert_torch", "ai_edge_torch"):
        try:
            mod = __import__(name)
        except ImportError:
            continue
        if hasattr(mod, "convert"):
            converter_mod = mod
            break
    if converter_mod is None:
        print("  [info] litert-torch / ai-edge-torch not available; trying ONNX fallback.")
        return False
    try:
        sample = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        edge = converter_mod.convert(model.eval(), (sample,))
        edge.export(str(out_path))
        print(f"  {converter_mod.__name__} wrote {out_path}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] {converter_mod.__name__} conversion failed: {e}")
        return False


def export_with_onnx(model: nn.Module, out_path: Path) -> bool:
    try:
        import onnx  # noqa: F401
        import tensorflow as tf
        import onnx_tf.backend as onnx_tf_backend
    except ImportError as e:
        print(f"  [error] ONNX fallback also unavailable: {e}")
        return False

    onnx_path = out_path.with_suffix(".onnx")
    saved_model_dir = out_path.with_suffix(".saved_model")
    sample = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model, sample, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        opset_version=14,
    )
    tf_rep = onnx_tf_backend.prepare(onnx.load(str(onnx_path)))
    tf_rep.export_graph(str(saved_model_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    out_path.write_bytes(converter.convert())
    print(f"  ONNX→TFLite wrote {out_path}")
    return True


def parity_check(model: nn.Module, tflite_path: Path) -> None:
    # litert-torch and tensorflow both register LLVM CLI options at import
    # time, so loading TF after conversion crashes the process. Run the
    # parity check in a subprocess that only imports the TF runtime, and
    # pipe the PyTorch logits in via JSON.
    rng = np.random.default_rng(0)
    sample_np = rng.standard_normal((1, 3, INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    with torch.no_grad():
        pt_logits = model(torch.from_numpy(sample_np)).numpy()[0]

    payload = {
        "tflite_path": str(tflite_path),
        "input_shape": list(sample_np.shape),
        "input_flat": sample_np.flatten().tolist(),
        "pt_logits": pt_logits.tolist(),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _PARITY_WORKER_SRC],
        input=json.dumps(payload).encode(),
        capture_output=True,
        check=False,
    )
    sys.stdout.write(proc.stdout.decode())
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode())
        print("  [warn] parity check could not run.")


_PARITY_WORKER_SRC = r"""
import json
import sys
import numpy as np

try:
    import ai_edge_litert.interpreter as lr
    Interpreter = lr.Interpreter
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        print("  [warn] no TFLite runtime available; skipping parity check.")
        sys.exit(0)

payload = json.loads(sys.stdin.read())
sample = np.array(payload['input_flat'], dtype=np.float32).reshape(payload['input_shape'])
pt_logits = np.array(payload['pt_logits'], dtype=np.float32)

interp = Interpreter(model_path=payload['tflite_path'])
interp.allocate_tensors()
in_detail = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]
in_shape = list(in_detail['shape'])
inp = sample if in_shape[1] == 3 else np.transpose(sample, (0, 2, 3, 1))
interp.set_tensor(in_detail['index'], inp.astype(in_detail['dtype']))
interp.invoke()
tf_logits = interp.get_tensor(out_detail['index'])[0]

pt_top = int(pt_logits.argmax())
tf_top = int(tf_logits.argmax())
max_abs_diff = float(np.max(np.abs(pt_logits - tf_logits)))
print(f"  PyTorch top-1 = {pt_top}")
print(f"  TFLite top-1  = {tf_top}")
print(f"  max |Delta logit| = {max_abs_diff:.4f}")
print("  Parity OK." if pt_top == tf_top and max_abs_diff < 0.5
      else "  [warn] parity diverges - inspect conversion.")
"""


def main() -> int:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Downloading {REPO_ID}/{WEIGHTS_FILENAME} (public, no token needed)")
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS_FILENAME)
    print(f"  cached at {weights_path}")

    print(f"[2/4] Building EfficientNet-B3 with {NUM_CLASSES}-class head")
    model = build_model()
    load_state_dict(model, weights_path)

    out_tflite = ASSETS_DIR / "crop_disease_model.tflite"
    print(f"[3/4] Exporting to {out_tflite}")
    ok = export_with_ai_edge_torch(model, out_tflite) or export_with_onnx(model, out_tflite)
    if not ok:
        print("[fatal] No converter succeeded. Install ai-edge-torch or onnx+onnx-tf+tensorflow.")
        return 1

    labels_path = ASSETS_DIR / "crop_disease_labels.txt"
    labels_path.write_text("\n".join(CLASS_LABELS) + "\n")
    print(f"  wrote {labels_path}")

    print("[4/4] Parity check (PyTorch vs TFLite, random tensor)")
    parity_check(model, out_tflite)

    size_mb = out_tflite.stat().st_size / (1024 * 1024)
    print(f"\nDone. {out_tflite.name} is {size_mb:.1f} MB. "
          "Rebuild the app to bundle it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
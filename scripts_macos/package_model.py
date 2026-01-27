# scripts_macos/package_model.py
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("phrase", nargs="?", default=None)
parser.add_argument("--phrase", dest="wake", default="hey_norman")
parser.add_argument("--lang", default="en", choices=["en", "ru"])
parser.add_argument("--id", dest="safe_id", default="")
args = parser.parse_args()

wake = args.phrase or args.wake
lang = args.lang

if args.safe_id:
    safe_id = re.sub(r"[^a-z0-9_]+", "", re.sub(r"\s+", "_", args.safe_id.lower()))
else:
    safe_id = re.sub(r"[^a-z0-9_]+", "", re.sub(r"\s+", "_", wake.lower()))
if not safe_id:
    h = hashlib.sha1(wake.encode("utf-8")).hexdigest()[:8]
    safe_id = f"wakeword_{h}"
src = Path("trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite")
dst = Path(f"{safe_id}.tflite")
if not src.exists():
    raise SystemExit(f"❌ Model not found at {src}")

shutil.copy(src, dst)

meta = {
  "type": "micro",
  "wake_word": wake,
  "author": "master phooey",
  "website": "https://github.com/MasterPhooey/MicroWakeWord-Trainer-Docker",
  "model": dst.name,
  "trained_languages": [lang],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.97,
    "sliding_window_size": 5,
    "feature_step_size": 10,
    "tensor_arena_size": 30000,
    "minimum_esphome_version": "2024.7.0"
  }
}
Path(f"{safe_id}.json").write_text(json.dumps(meta, indent=2))
print(f"📦 Wrote {dst.name} and {safe_id}.json")

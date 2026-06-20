# Running Prototype 0 on Google Colab

This run checks that the activation-capture and steering apparatus transfers
from Qwen3-0.6B-Base to Qwen3-1.7B-Base. It is still an instrumentation test,
not an emotion-representation result.

## 1. Start a GPU runtime

In Colab, select **Runtime → Change runtime type → T4 GPU**. Then verify it:

```python
import torch

assert torch.cuda.is_available(), "Select a GPU runtime before continuing"
print(torch.cuda.get_device_name(0))
```

## 2. Clone and install

Replace the branch with `main` after this change has been merged.

```python
!git clone --branch codex/colab-1-7b-compat \
  https://github.com/prayush21/functional-emotions.git
%cd functional-emotions
!python -m pip install -q -e ".[dev]"
```

The model is public, so a Hugging Face token is optional. Authentication can
improve rate limits for repeated downloads.

## 3. Test the apparatus code

```python
!pytest -q
```

Expected: all tests pass without downloading model weights.

## 4. Run the 1.7B compatibility experiment

```python
!fe-prototype0 --config configs/prototype0_qwen3_1.7b_colab.yaml
```

The first run downloads Qwen3-1.7B-Base. Colab caches the weights only for the
life of the runtime unless Google Drive or another persistent cache is mounted.

## 5. Inspect the latest result

```python
import json
from pathlib import Path

root = Path("artifacts/prototype0-qwen3-1.7b-colab")
latest = max((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
metrics = json.loads((latest / "metrics.json").read_text())

print("artifact directory:", latest)
print("all gates passed:", metrics["all_hard_gates_pass"])
print("gates:", metrics["gates"])
print("layer:", metrics["layer_index"])
print("hidden size:", metrics["hidden_size"])
print("effect monotonicity:", metrics["effect_monotonic_spearman"])
```

For this model, expect 28 layers, middle layer 14, and a hidden size of 2,048.
The exact steering metrics may differ from the 0.6B model. The important
compatibility criteria are the three hard gates, not identical happy/sad scores.

## 6. Download the complete artifact bundle

```python
import shutil
from google.colab import files

archive = shutil.make_archive("/content/prototype0-qwen3-1.7b-results", "zip", latest)
files.download(archive)
```


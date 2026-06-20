"""Run Prototype 0 on a short-lived Modal GPU.

Invoke from the repository root with: modal run cloud/modal_prototype0.py
"""

from pathlib import Path

import modal


app = modal.App("functional-emotions-prototype0")
repository = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.26,<3",
        "pyyaml>=6,<7",
        "safetensors>=0.4,<1",
        "torch>=2.4,<3",
        "transformers>=4.51,<6",
    )
    .add_local_dir(repository / "src", remote_path="/root/src")
    .add_local_dir(repository / "configs", remote_path="/root/configs")
)
model_cache = modal.Volume.from_name("functional-emotions-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=30 * 60,
    volumes={"/root/.cache/huggingface": model_cache},
    env={"PYTHONPATH": "/root/src"},
)
def run() -> dict:
    import json
    import sys

    sys.path.insert(0, "/root/src")
    from functional_emotions.prototype0 import load_config, run as run_experiment

    config = load_config(Path("/root/configs/prototype0.yaml"))
    config["model"]["name"] = "Qwen/Qwen3-1.7B-Base"
    config["model"]["dtype"] = "float16"
    config["model"]["device"] = "cuda"
    config["output_dir"] = "/tmp/prototype0"
    output = run_experiment(config)
    return json.loads((output / "metrics.json").read_text())


@app.local_entrypoint()
def main() -> None:
    print(run.remote())

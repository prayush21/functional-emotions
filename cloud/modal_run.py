"""Run any experiment prototype on a short-lived Modal GPU.

Supersedes ``cloud/modal_prototype0.py`` with a parameterized entrypoint:

    modal run cloud/modal_run.py --prototype 51 --config configs/prototype51.yaml --gpu T4

Artifacts land on the ``functional-emotions-artifacts`` volume so they can be
fetched locally with ``cloud/fetch_run.py``. Input bundles referenced by a
config (``results/runs/...``) are shipped in the image because ``results/`` is
small and version-controlled; see README "Colab and Modal".
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prototype_registry import (  # noqa: E402  (local module, added to sys.path above)
    artifact_output_dir,
    resolve_spec,
    resolve_stage,
    volume_subdir,
)

app = modal.App("functional-emotions-run")
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
    .add_local_dir(repository / "results", remote_path="/root/results")
    # The app module is re-imported inside the container at /root/modal_run.py,
    # so its sibling import must resolve from /root as well.
    .add_local_file(
        repository / "cloud" / "prototype_registry.py",
        remote_path="/root/prototype_registry.py",
    )
)
model_cache = modal.Volume.from_name("functional-emotions-hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("functional-emotions-artifacts", create_if_missing=True)

ARTIFACTS_ROOT = "/artifacts"


def _dispatch(module_name: str, stage: str | None, config: dict, run_dir: str | None) -> Path:
    """Import the prototype module and call its run/stage entrypoint(s)."""

    import importlib

    module = importlib.import_module(module_name)

    if module_name.endswith("prototype25"):
        run_path = Path(run_dir) if run_dir else None
        return module.run(config, stage or "prepare", run_path)

    if module_name.endswith("prototype1"):
        chosen = stage or "extract"
        output: Path | None = None
        if chosen in {"generate", "all"}:
            output = module.generate_dataset(config)
        if chosen in {"extract", "all"}:
            output = module.run(config)
        assert output is not None
        return output

    return module.run(config)


@app.function(
    image=image,
    timeout=60 * 60,
    # Long runs get preempted (observed repeatedly on L4 and A100, 2026-07-11);
    # retries re-schedule the input server-side so a detached run survives both
    # preemption and local-client disconnects. Prototype runners create a fresh
    # timestamped output directory per attempt, so retries cannot collide.
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={"/root/.cache/huggingface": model_cache, ARTIFACTS_ROOT: artifacts},
    env={"PYTHONPATH": "/root/src"},
)
def run_prototype(
    module_name: str,
    config_path: str,
    stage: str | None,
    run_dir: str | None,
) -> dict:
    import json

    sys.path.insert(0, "/root/src")
    from functional_emotions.prototype0 import load_config

    config = load_config(Path("/root") / config_path)
    config["output_dir"] = artifact_output_dir(config, ARTIFACTS_ROOT)

    output = _dispatch(module_name, stage, config, run_dir)

    artifacts.commit()

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return {
        "run_dir": output.name,
        "output_dir": str(output),
        "summary": summary,
    }


@app.local_entrypoint()
def main(
    prototype: str,
    config: str,
    gpu: str = "T4",
    stage: str | None = None,
    run_dir: str | None = None,
    timeout_minutes: int = 240,
    spawn: bool = False,
) -> None:
    import json

    spec = resolve_spec(prototype)
    chosen_stage = resolve_stage(spec, stage)

    function = run_prototype.with_options(gpu=gpu, timeout=timeout_minutes * 60)
    kwargs = {
        "module_name": spec.module,
        "config_path": config,
        "stage": chosen_stage,
        "run_dir": run_dir,
    }
    if spawn:
        # .remote() inputs are cancelled if the local client disconnects, even
        # under `modal run --detach` (observed repeatedly on multi-hour runs).
        # .spawn() decouples the call: pair with --detach, then poll the
        # artifacts volume / `modal app list`, and fetch with cloud/fetch_run.py.
        handle = function.spawn(**kwargs)
        print(
            json.dumps(
                {
                    "spawned_function_call_id": handle.object_id,
                    "prototype": spec.key,
                    "artifacts_subdir": volume_subdir(prototype),
                },
                indent=2,
            )
        )
        return
    result = function.remote(**kwargs)
    result["prototype"] = spec.key
    result["artifacts_subdir"] = volume_subdir(prototype)
    print(json.dumps(result, indent=2))

from __future__ import annotations

import argparse
import copy
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml

from . import prototype1, prototype2
from .tracking import build_manifest, git_metadata, make_run_id, sha256_json


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def resolved_prototype1_config(config: dict[str, Any], output: Path) -> dict[str, Any]:
    base_path = Path(config["prototype1"].get("base_config", "configs/prototype1.yaml"))
    base = load_config(base_path)
    overrides = config["prototype1"].get("overrides", {})
    resolved = deep_merge(base, overrides)
    resolved["experiment"] = "prototype1-prototype25"
    resolved["seed"] = int(config.get("seed", resolved.get("seed", 42)))
    resolved["data"]["stories_path"] = str(output / "dataset" / "stories.jsonl")
    resolved["data"]["neutral_path"] = str(output / "dataset" / "neutral.jsonl")
    resolved["output_dir"] = str(output / "prototype1")
    return resolved


def resolved_prototype2_config(
    config: dict[str, Any],
    output: Path,
    prototype1_run_dir: Path | None = None,
) -> dict[str, Any]:
    base_path = Path(config["prototype2"].get("base_config", "configs/prototype2.yaml"))
    base = load_config(base_path)
    overrides = config["prototype2"].get("overrides", {})
    resolved = deep_merge(base, overrides)
    resolved["experiment"] = "prototype2-prototype25-validation"
    resolved["seed"] = int(config.get("seed", resolved.get("seed", 42)))
    resolved["output_dir"] = str(output / "prototype2")
    if prototype1_run_dir is not None:
        resolved["prototype1"]["run_dir"] = str(prototype1_run_dir)
    elif config["prototype2"].get("prototype1_run_dir"):
        resolved["prototype1"]["run_dir"] = str(config["prototype2"]["prototype1_run_dir"])
    else:
        resolved["prototype1"]["run_dir"] = str(output / "prototype1" / "SET_AFTER_EXTRACT")
    return resolved


def dataset_plan(config: dict[str, Any]) -> dict[str, Any]:
    p1 = config["prototype1"]["overrides"]
    data = p1["data"]
    generation = p1["generation"]
    emotions = list(data["emotions"])
    topics = list(data["topics"])
    stories_per_pair = int(generation["stories_per_topic_emotion"])
    return {
        "emotions": emotions,
        "topics": len(topics),
        "neutral_topics": len(data["neutral_topics"]),
        "stories_per_topic_emotion": stories_per_pair,
        "total_emotion_stories": len(emotions) * len(topics) * stories_per_pair,
        "train_topic_fraction": data["train_topic_fraction"],
        "purpose": (
            "Revise Prototype 1 extraction/data before geometry: more topics, "
            "more stories per pair, and balanced negative-emotion premises."
        ),
    }


def complete_prototype1_runs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    required = (
        "config.json",
        "metrics.json",
        "emotion_vectors.safetensors",
        "neutral_pca.safetensors",
        "dataset/stories.jsonl",
    )
    return sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_dir() and all((candidate / name).is_file() for name in required)
    )


def create_output(config: dict[str, Any], run_dir: Path | None = None) -> tuple[Path, str, str]:
    created_at = datetime.now(timezone.utc)
    if run_dir is not None:
        output = run_dir
        run_id = output.name
        output.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
        config_hash = sha256_json(config)
        model_name = config["prototype1"]["overrides"]["model"]["name"]
        run_id = make_run_id(
            timestamp=timestamp,
            experiment=config["experiment"],
            model_name=model_name,
            seed=int(config["seed"]),
            config_hash=config_hash,
        )
        output = Path(config["output_dir"]) / run_id
        output.mkdir(parents=True, exist_ok=False)
    return output, run_id, created_at.isoformat()


def write_orchestrator_files(
    *,
    output: Path,
    run_id: str,
    created_at: str,
    config: dict[str, Any],
    prototype1_config: dict[str, Any],
    prototype2_config: dict[str, Any],
    stage: str,
    prototype1_run_dir: Path | None = None,
    prototype2_run_dir: Path | None = None,
) -> None:
    write_yaml(output / "prototype1_config.yaml", prototype1_config)
    write_yaml(output / "prototype2_config.yaml", prototype2_config)
    code = git_metadata()
    metrics = {
        "stage": stage,
        "dataset_plan": dataset_plan(config),
        "prototype1_run_dir": str(prototype1_run_dir) if prototype1_run_dir else None,
        "prototype2_run_dir": str(prototype2_run_dir) if prototype2_run_dir else None,
        "next_decision_rule": {
            "proceed_to_prototype3_if": [
                "Prototype 2.5 validation beats shuffled labels",
                "lexical robustness exceeds chance",
                "mean intensity Spearman is positive",
                "angry dominance and afraid argmax failure are materially reduced",
            ],
            "otherwise": "Revise generation/extraction again before geometry",
        },
    }
    environment = {
        "created_at": created_at,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "code_git_commit": code["commit"],
        "code_git_dirty": code["dirty"],
        "run_id": run_id,
        "config_sha256": sha256_json(config),
    }
    manifest = build_manifest(
        run_id=run_id,
        created_at=created_at,
        config={
            "experiment": config["experiment"],
            "seed": config["seed"],
            "model": prototype1_config["model"],
        },
        resolved_model_revision=None,
        code=code,
    )
    summary = {
        "stage": stage,
        "dataset_plan": metrics["dataset_plan"],
        "prototype1_run_dir": metrics["prototype1_run_dir"],
        "prototype2_run_dir": metrics["prototype2_run_dir"],
    }
    for filename, value in (
        ("config.json", config),
        ("metrics.json", metrics),
        ("environment.json", environment),
        ("manifest.json", manifest),
        ("summary.json", summary),
    ):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n")


def run(config: dict[str, Any], stage: str, run_dir: Path | None = None) -> Path:
    output, run_id, created_at = create_output(config, run_dir)
    p1_config = resolved_prototype1_config(config, output)
    p2_config = resolved_prototype2_config(config, output)
    prototype1_run_dir = None
    prototype2_run_dir = None

    write_orchestrator_files(
        output=output,
        run_id=run_id,
        created_at=created_at,
        config=config,
        prototype1_config=p1_config,
        prototype2_config=p2_config,
        stage="prepare",
    )
    if stage == "prepare":
        print(json.dumps({"output": str(output), "stage": "prepare"}, indent=2))
        return output

    if stage in {"generate", "all"}:
        prototype1.generate_dataset(p1_config)

    if stage in {"extract", "all"}:
        prototype1_run_dir = prototype1.run(p1_config)

    if stage == "validate":
        configured = config["prototype2"].get("prototype1_run_dir")
        if configured:
            prototype1_run_dir = Path(configured)
        else:
            candidates = complete_prototype1_runs(output / "prototype1")
            if not candidates:
                raise ValueError(
                    "prototype2.prototype1_run_dir is required for --stage validate "
                    "unless --run-dir contains a completed nested Prototype 1 run"
                )
            prototype1_run_dir = candidates[-1]

    if stage in {"validate", "all"}:
        if prototype1_run_dir is None:
            raise ValueError("Prototype 1 run directory is required before validation")
        p2_config = resolved_prototype2_config(config, output, prototype1_run_dir)
        prototype2_run_dir = prototype2.run(p2_config)

    write_orchestrator_files(
        output=output,
        run_id=run_id,
        created_at=created_at,
        config=config,
        prototype1_config=p1_config,
        prototype2_config=p2_config,
        stage=stage,
        prototype1_run_dir=prototype1_run_dir,
        prototype2_run_dir=prototype2_run_dir,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "stage": stage,
                "prototype1_run_dir": str(prototype1_run_dir) if prototype1_run_dir else None,
                "prototype2_run_dir": str(prototype2_run_dir) if prototype2_run_dir else None,
            },
            indent=2,
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prototype 2.5 revised extraction pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/prototype25.yaml"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Reuse an existing Prototype 2.5 orchestrator directory for staged runs.",
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "generate", "extract", "validate", "all"),
        default="prepare",
    )
    args = parser.parse_args()
    run(load_config(args.config), args.stage, args.run_dir)


if __name__ == "__main__":
    main()

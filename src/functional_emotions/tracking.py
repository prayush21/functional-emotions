from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
REQUIRED_BUNDLE_FILES = ("config.json", "environment.json", "metrics.json")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "unknown"


def git_metadata(cwd: Path | None = None) -> dict[str, Any]:
    root = cwd or Path.cwd()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def timestamp_from_iso(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(
    *, timestamp: str, experiment: str, model_name: str, seed: int, config_hash: str
) -> str:
    return (
        f"{timestamp}__{slugify(experiment)}__{slugify(model_name)}"
        f"__seed-{seed}__{config_hash[:10]}"
    )


def summary_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    if "selected_layer" in metrics:
        selected = next(
            (
                row["pca_cleaned"]
                for row in metrics.get("layers", [])
                if row.get("layer") == metrics["selected_layer"]
            ),
            {},
        )
        return {
            "all_hard_gates_pass": metrics.get("all_hard_gates_pass"),
            "selected_layer": metrics.get("selected_layer"),
            "held_out_accuracy": selected.get("accuracy"),
            "held_out_macro_auc": selected.get("macro_auc"),
            "mean_correct_margin": selected.get("mean_correct_margin"),
            "number_of_emotions": len(selected.get("per_emotion", {})),
        }
    sweep = metrics.get("sweep", [])
    zero = next((row for row in sweep if row.get("raw_strength") == 0.0), None)
    negative = min(sweep, key=lambda row: row["raw_strength"]) if sweep else None
    positive = max(sweep, key=lambda row: row["raw_strength"]) if sweep else None
    return {
        "all_hard_gates_pass": metrics.get("all_hard_gates_pass"),
        "layer_index": metrics.get("layer_index"),
        "hidden_size": metrics.get("hidden_size"),
        "effect_monotonic_spearman": metrics.get("effect_monotonic_spearman"),
        "baseline_margin": (zero.get("positive_minus_negative_logit_margin") if zero else None),
        "minimum_strength": negative.get("raw_strength") if negative else None,
        "minimum_strength_margin": (
            negative.get("positive_minus_negative_logit_margin") if negative else None
        ),
        "maximum_strength": positive.get("raw_strength") if positive else None,
        "maximum_strength_margin": (
            positive.get("positive_minus_negative_logit_margin") if positive else None
        ),
    }


def build_manifest(
    *,
    run_id: str,
    created_at: str,
    config: dict[str, Any],
    resolved_model_revision: str | None,
    code: dict[str, Any],
    legacy_import: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "experiment": config["experiment"],
        "seed": config["seed"],
        "config_sha256": sha256_json(config),
        "model": {
            "name": config["model"]["name"],
            "requested_revision": config["model"].get("revision"),
            "resolved_revision": resolved_model_revision,
        },
        "code": code,
        "legacy_import": legacy_import,
    }


def load_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (bundle / name).is_file()]
    if missing:
        raise ValueError(f"Missing required bundle files: {', '.join(missing)}")
    return tuple(json.loads((bundle / name).read_text()) for name in REQUIRED_BUNDLE_FILES)  # type: ignore[return-value]


def register_bundle(bundle: Path, registry: Path = Path("results")) -> Path:
    config, environment, metrics = load_bundle(bundle)
    created_at = environment["created_at"]
    config_hash = sha256_json(config)
    existing_manifest_path = bundle / "manifest.json"
    if existing_manifest_path.is_file():
        manifest = json.loads(existing_manifest_path.read_text())
        run_id = manifest["run_id"]
    else:
        run_id = make_run_id(
            timestamp=timestamp_from_iso(created_at),
            experiment=config["experiment"],
            model_name=config["model"]["name"],
            seed=int(config["seed"]),
            config_hash=config_hash,
        )
        manifest = build_manifest(
            run_id=run_id,
            created_at=created_at,
            config=config,
            resolved_model_revision=environment.get("resolved_model_revision"),
            code={
                "commit": environment.get("code_git_commit"),
                "dirty": environment.get("code_git_dirty"),
            },
            legacy_import=True,
        )

    destination = registry / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_BUNDLE_FILES:
        shutil.copy2(bundle / name, destination / name)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = summary_from_metrics(metrics)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    index_path = registry / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if index_path.exists():
        records = [json.loads(line) for line in index_path.read_text().splitlines() if line]
    if not any(record["run_id"] == run_id for record in records):
        record = {
            "run_id": run_id,
            "created_at": created_at,
            "experiment": config["experiment"],
            "model_name": config["model"]["name"],
            "resolved_model_revision": manifest["model"]["resolved_revision"],
            "code_commit": manifest["code"]["commit"],
            **summary,
            "path": str(destination),
        }
        with index_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return destination

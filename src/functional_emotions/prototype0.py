from __future__ import annotations

import argparse
import json
import math
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from safetensors.torch import save_file
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .instrumentation import (
    capture_layer_output,
    decoder_layers,
    resolve_layer_index,
    steer_layer_output,
)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device in {"cuda", "mps"} else torch.float32
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {name}")
    return dtype


def encode(tokenizer: Any, texts: list[str], device: str) -> dict[str, Tensor]:
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return {key: value.to(device) for key, value in batch.items()}


def last_token_activations(
    model: Any, tokenizer: Any, layer: Any, prompts: list[str], device: str
) -> Tensor:
    batch = encode(tokenizer, prompts, device)
    with torch.inference_mode(), capture_layer_output(layer) as captured:
        model(**batch)
    hidden = captured[0]
    indices = batch["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, indices].float().cpu()


def derive_direction(positive: Tensor, negative: Tensor) -> Tensor:
    direction = positive.mean(dim=0) - negative.mean(dim=0)
    norm = direction.norm()
    if not torch.isfinite(norm) or norm.item() == 0:
        raise ValueError("Calibration prompts produced a zero or non-finite direction")
    return direction / norm


def next_token_logits(model: Any, inputs: dict[str, Tensor]) -> Tensor:
    with torch.inference_mode():
        return model(**inputs).logits[:, -1, :].float().cpu()


def steered_logits(
    model: Any,
    layer: Any,
    inputs: dict[str, Tensor],
    direction: Tensor,
    strength: float,
) -> Tensor:
    with torch.inference_mode(), steer_layer_output(layer, direction, strength):
        return model(**inputs).logits[:, -1, :].float().cpu()


def kl_from_logits(baseline: Tensor, candidate: Tensor) -> float:
    base_logp = baseline.log_softmax(dim=-1)
    candidate_logp = candidate.log_softmax(dim=-1)
    base_p = base_logp.exp()
    return float((base_p * (base_logp - candidate_logp)).sum(dim=-1).mean())


def token_id(tokenizer: Any, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2 or len(set(values_a)) < 2 or len(set(values_b)) < 2:
        return math.nan
    return float(np.corrcoef(rankdata(values_a), rankdata(values_b))[0, 1])


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def run(config: dict[str, Any]) -> Path:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = choose_device(config["model"]["device"])
    dtype = choose_dtype(config["model"]["dtype"], device)
    model_name = config["model"]["name"]
    revision = config["model"]["revision"]
    local_only = bool(config["model"].get("local_files_only", False))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, local_files_only=local_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        local_files_only=local_only,
    ).to(device)
    model.eval()

    layers = decoder_layers(model)
    layer_index = resolve_layer_index(config["model"]["layer"], len(layers))
    layer = layers[layer_index]

    positive = last_token_activations(
        model, tokenizer, layer, config["calibration"]["positive"], device
    )
    negative = last_token_activations(
        model, tokenizer, layer, config["calibration"]["negative"], device
    )
    direction = derive_direction(positive, negative)

    prompt_inputs = encode(tokenizer, [config["steering"]["prompt"]], device)
    baseline = next_token_logits(model, prompt_inputs)
    baseline_activation = last_token_activations(
        model, tokenizer, layer, [config["steering"]["prompt"]], device
    )[0]
    baseline_norm = float(baseline_activation.norm())

    positive_id = token_id(tokenizer, config["steering"]["target_tokens"]["positive"])
    negative_id = token_id(tokenizer, config["steering"]["target_tokens"]["negative"])

    rows: list[dict[str, Any]] = []
    for raw_strength in config["steering"]["strengths"]:
        raw_strength = float(raw_strength)
        applied_strength = raw_strength
        if config["steering"].get("scale_by_baseline_norm", True):
            applied_strength *= baseline_norm
        logits = steered_logits(model, layer, prompt_inputs, direction.to(device), applied_strength)
        difference = logits - baseline
        margin = None
        if positive_id is not None and negative_id is not None:
            margin = float(logits[0, positive_id] - logits[0, negative_id])
        rows.append(
            {
                "raw_strength": raw_strength,
                "applied_strength": applied_strength,
                "max_abs_logit_change": float(difference.abs().max()),
                "logit_l2": float(difference.norm()),
                "kl_from_baseline": kl_from_logits(baseline, logits),
                "positive_minus_negative_logit_margin": margin,
            }
        )

    zero_row = next(row for row in rows if row["raw_strength"] == 0.0)
    nonzero_rows = [row for row in rows if row["raw_strength"] != 0.0]
    absolute_strengths = [abs(row["raw_strength"]) for row in nonzero_rows]
    effect_sizes = [row["logit_l2"] for row in nonzero_rows]
    monotonicity = spearman(absolute_strengths, effect_sizes)

    thresholds = config["gates"]
    gates = {
        "zero_intervention_reproduces_baseline": (
            zero_row["max_abs_logit_change"]
            <= float(thresholds["zero_steering_max_abs_logit_error"])
        ),
        "nonzero_intervention_changes_logits": (
            max(effect_sizes) >= float(thresholds["minimum_nonzero_logit_l2"])
        ),
        "effect_grows_with_absolute_strength": (
            monotonicity >= float(thresholds["minimum_effect_monotonic_spearman"])
        ),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(config["output_dir"]) / timestamp
    output.mkdir(parents=True, exist_ok=False)
    resolved = json.loads(json.dumps(config))
    resolved["model"]["resolved_device"] = device
    resolved["model"]["resolved_dtype"] = str(dtype)
    resolved["model"]["resolved_layer_index"] = layer_index
    resolved["model"]["number_of_layers"] = len(layers)

    metrics = {
        "all_hard_gates_pass": all(gates.values()),
        "gates": gates,
        "layer_index": layer_index,
        "hidden_size": int(direction.numel()),
        "baseline_activation_norm": baseline_norm,
        "direction_norm": float(direction.norm()),
        "effect_monotonic_spearman": monotonicity,
        "target_token_ids": {"positive": positive_id, "negative": negative_id},
        "sweep": rows,
    }
    environment = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "model_name": model_name,
        "requested_revision": revision,
    }

    for filename, value in (
        ("config.json", resolved),
        ("metrics.json", metrics),
        ("environment.json", environment),
    ):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n")
    save_file({"direction": direction.contiguous()}, output / "direction.safetensors")

    print(json.dumps({"output": str(output), **metrics}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prototype 0 instrumentation experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/prototype0.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

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
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .instrumentation import decoder_layers, resolve_layer_index
from .prototype0 import choose_device, choose_dtype, encode, next_token_logits
from .prototype4 import (
    apply_steering_scale,
    default_model_config,
    kl_from_logits,
    load_selected_emotion_vectors,
    matched_random_vector,
    resolve_artifact_path,
    resolve_selected_layer,
    spearman,
    steered_next_token_logits,
    wrong_emotion,
)
from .tracking import build_manifest, git_metadata, make_run_id, sha256_json


INTERPRETATION_CAVEAT = (
    "These are compact four-emotion vectors with weak valence/arousal geometry. "
    "Prototype 5 tests whether local causal steering transfers to simple preference "
    "behavior, not whether the model has mature human-like emotion structure."
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def preference_prompt(activity_a: str, activity_b: str) -> str:
    return (
        "The person is deciding what to do next.\n"
        f"A: {activity_a}\n"
        f"B: {activity_b}\n"
        "The better next action is:"
    )


def parse_activity_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for index, row in enumerate(rows):
        option_a = row.get("activity_a") or row.get("a")
        option_b = row.get("activity_b") or row.get("b")
        category_a = row.get("category_a")
        category_b = row.get("category_b")
        if not option_a or not option_b or not category_a or not category_b:
            raise ValueError(
                "Each activity pair needs activity_a, activity_b, category_a, and category_b"
            )
        pairs.append(
            {
                "id": row.get("id") or f"pair-{index + 1}",
                "activity_a": str(option_a),
                "activity_b": str(option_b),
                "category_a": str(category_a),
                "category_b": str(category_b),
                "prompt": row.get("prompt") or preference_prompt(str(option_a), str(option_b)),
            }
        )
    if not pairs:
        raise ValueError("At least one activity pair is required")
    return pairs


def choice_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    return {
        "A": token_ids_for_choice(tokenizer, "A"),
        "B": token_ids_for_choice(tokenizer, "B"),
    }


def token_ids_for_choice(tokenizer: Any, label: str) -> list[int]:
    ids = []
    for variant in (label, f" {label}"):
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if encoded:
            ids.append(int(encoded[0]))
    return sorted(set(ids))


def mean_choice_logit(logits: Tensor, token_ids: list[int]) -> Tensor:
    if not token_ids:
        return torch.full((logits.shape[0],), math.nan)
    return logits[:, token_ids].mean(dim=1)


def preference_margin(logits: Tensor, token_ids: dict[str, list[int]]) -> Tensor:
    return mean_choice_logit(logits, token_ids["A"]) - mean_choice_logit(
        logits, token_ids["B"]
    )


def expected_direction(
    emotion: str,
    category_a: str,
    category_b: str,
    expected_categories: dict[str, list[str]],
) -> int:
    targets = set(expected_categories.get(emotion, []))
    a_matches = category_a in targets
    b_matches = category_b in targets
    if a_matches and not b_matches:
        return 1
    if b_matches and not a_matches:
        return -1
    return 0


def preference_score_row(
    *,
    pair: dict[str, Any],
    target_emotion: str,
    vector_label: str,
    control: str,
    raw_strength: float,
    applied_strength: float,
    baseline_logits: Tensor,
    logits: Tensor,
    token_ids: dict[str, list[int]],
    expected_categories: dict[str, list[str]],
) -> dict[str, Any]:
    baseline_margin = float(preference_margin(baseline_logits, token_ids)[0])
    margin = float(preference_margin(logits, token_ids)[0])
    delta = margin - baseline_margin
    direction = expected_direction(
        target_emotion,
        pair["category_a"],
        pair["category_b"],
        expected_categories,
    )
    expected_effect = float(direction * delta) if direction else math.nan
    return {
        "pair_id": pair["id"],
        "prompt": pair["prompt"],
        "activity_a": pair["activity_a"],
        "activity_b": pair["activity_b"],
        "category_a": pair["category_a"],
        "category_b": pair["category_b"],
        "target_emotion": target_emotion,
        "vector_label": vector_label,
        "control": control,
        "raw_strength": raw_strength,
        "applied_strength": applied_strength,
        "baseline_margin_a_minus_b": baseline_margin,
        "margin_a_minus_b": margin,
        "margin_delta_from_baseline": delta,
        "expected_direction": direction,
        "expected_effect": expected_effect,
        "matches_hypothesis": direction != 0,
        "max_abs_logit_change": float((logits - baseline_logits).abs().max()),
        "kl_from_baseline": kl_from_logits(baseline_logits, logits),
    }


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return float(np.mean(finite)) if finite else math.nan


def aggregate_preference_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["target_emotion"], row["control"], float(row["raw_strength"]))
        grouped.setdefault(key, []).append(row)
    output = []
    for (emotion, control, strength), group in sorted(grouped.items()):
        expected_values = [row["expected_effect"] for row in group]
        matched = [row["expected_effect"] for row in group if row["matches_hypothesis"]]
        nonmatched_abs = [
            abs(row["margin_delta_from_baseline"])
            for row in group
            if not row["matches_hypothesis"]
        ]
        output.append(
            {
                "target_emotion": emotion,
                "control": control,
                "raw_strength": strength,
                "mean_margin_delta": float(
                    np.mean([row["margin_delta_from_baseline"] for row in group])
                ),
                "mean_expected_effect": finite_mean(expected_values),
                "mean_matching_expected_effect": finite_mean(matched),
                "mean_nonmatching_abs_delta": finite_mean(nonmatched_abs),
                "mean_kl_from_baseline": float(
                    np.mean([row["kl_from_baseline"] for row in group])
                ),
                "max_abs_logit_change": float(
                    max(row["max_abs_logit_change"] for row in group)
                ),
            }
        )
    return {"rows": output}


def positive_strength_summary(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    real = [
        row
        for row in aggregates
        if row["control"] == "real" and float(row["raw_strength"]) > 0.0
    ]
    return {
        "mean_expected_effect": finite_mean(
            [row["mean_expected_effect"] for row in real]
        ),
        "mean_matching_expected_effect": finite_mean(
            [row["mean_matching_expected_effect"] for row in real]
        ),
        "mean_nonmatching_abs_delta": finite_mean(
            [row["mean_nonmatching_abs_delta"] for row in real]
        ),
    }


def control_summary(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for control in ("real", "random", "wrong_emotion"):
        rows = [
            row
            for row in aggregates
            if row["control"] == control and float(row["raw_strength"]) > 0.0
        ]
        summary[control] = {
            "mean_expected_effect": finite_mean(
                [row["mean_expected_effect"] for row in rows]
            ),
            "mean_matching_expected_effect": finite_mean(
                [row["mean_matching_expected_effect"] for row in rows]
            ),
        }
    return summary


def dose_response(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for row in aggregates
        if row["control"] == "real" and float(row["raw_strength"]) > 0.0
    ]
    if not rows:
        return {"spearman_by_emotion": {}, "mean_spearman": math.nan}
    by_emotion: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_emotion.setdefault(row["target_emotion"], []).append(row)
    correlations = {}
    for emotion, emotion_rows in by_emotion.items():
        correlations[emotion] = spearman(
            [float(row["raw_strength"]) for row in emotion_rows],
            [float(row["mean_expected_effect"]) for row in emotion_rows],
        )
    finite = [value for value in correlations.values() if not math.isnan(value)]
    return {
        "spearman_by_emotion": correlations,
        "mean_spearman": float(np.mean(finite)) if finite else math.nan,
    }


def run_preference_interventions(
    *,
    model: Any,
    tokenizer: Any,
    layer: Any,
    device: str,
    vectors: dict[str, Tensor],
    emotions: list[str],
    pairs: list[dict[str, Any]],
    strengths: list[float],
    token_ids: dict[str, list[int]],
    expected_categories: dict[str, list[str]],
    random_seed: int,
    scale_by_vector_norm: bool,
) -> dict[str, Any]:
    rows = []
    for pair in pairs:
        inputs = encode(tokenizer, [pair["prompt"]], device)
        baseline = next_token_logits(model, inputs)
        for target in emotions:
            controls = {
                "real": (target, vectors[target]),
                "random": (
                    target,
                    matched_random_vector(
                        vectors[target], random_seed + emotions.index(target)
                    ),
                ),
                "wrong_emotion": (
                    wrong_emotion(target, emotions),
                    vectors[wrong_emotion(target, emotions)],
                ),
            }
            for raw_strength in strengths:
                raw_strength = float(raw_strength)
                for control, (vector_label, vector) in controls.items():
                    applied = apply_steering_scale(
                        raw_strength, vector, scale_by_vector_norm
                    )
                    logits = steered_next_token_logits(
                        model, layer, inputs, vector.to(device), applied
                    )
                    rows.append(
                        preference_score_row(
                            pair=pair,
                            target_emotion=target,
                            vector_label=vector_label,
                            control=control,
                            raw_strength=raw_strength,
                            applied_strength=applied,
                            baseline_logits=baseline,
                            logits=logits,
                            token_ids=token_ids,
                            expected_categories=expected_categories,
                        )
                    )
    aggregates = aggregate_preference_rows(rows)["rows"]
    return {
        "choice_token_ids": token_ids,
        "expected_categories": expected_categories,
        "pairs": pairs,
        "rows": rows,
        "aggregates": aggregates,
        "positive_strength_summary": positive_strength_summary(aggregates),
        "control_summary": control_summary(aggregates),
        "dose_response": dose_response(aggregates),
    }


def zero_fidelity(preferences: dict[str, Any]) -> dict[str, Any]:
    zero_rows = [
        row for row in preferences["rows"] if float(row["raw_strength"]) == 0.0
    ]
    max_error = max((row["max_abs_logit_change"] for row in zero_rows), default=math.inf)
    return {"max_abs_logit_error": float(max_error), "rows": len(zero_rows)}


def soft_gates(preferences: dict[str, Any]) -> dict[str, bool]:
    aggregates = preferences["aggregates"]
    controls = preferences["control_summary"]
    dose = preferences["dose_response"]
    real_positive = controls["real"]["mean_expected_effect"]
    random_positive = controls["random"]["mean_expected_effect"]
    wrong_positive = controls["wrong_emotion"]["mean_expected_effect"]
    negative_real = [
        row
        for row in aggregates
        if row["control"] == "real" and float(row["raw_strength"]) < 0.0
    ]
    positive_real = [
        row
        for row in aggregates
        if row["control"] == "real" and float(row["raw_strength"]) > 0.0
    ]
    negative_mean = finite_mean([row["mean_expected_effect"] for row in negative_real])
    positive_mean = finite_mean([row["mean_expected_effect"] for row in positive_real])
    matching_mean = preferences["positive_strength_summary"][
        "mean_matching_expected_effect"
    ]
    nonmatching_mean = preferences["positive_strength_summary"][
        "mean_nonmatching_abs_delta"
    ]
    return {
        "positive_steering_shifts_expected_preferences": real_positive > 0.0,
        "opposite_sign_moves_opposite_direction": negative_mean < positive_mean,
        "intended_effect_exceeds_random_control": real_positive > random_positive,
        "intended_effect_exceeds_wrong_emotion_control": real_positive > wrong_positive,
        "dose_response_spearman_positive": dose["mean_spearman"] > 0.0,
        "matching_pairs_stronger_than_nonmatching_pairs": matching_mean
        > nonmatching_mean,
    }


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def update_elo(ratings: dict[str, float], category_a: str, category_b: str, margin: float) -> None:
    ratings.setdefault(category_a, 1000.0)
    ratings.setdefault(category_b, 1000.0)
    expected_a = 1.0 / (1.0 + 10 ** ((ratings[category_b] - ratings[category_a]) / 400.0))
    score_a = sigmoid(margin)
    delta = 16.0 * (score_a - expected_a)
    ratings[category_a] += delta
    ratings[category_b] -= delta


def elo_diagnostic(preferences: dict[str, Any]) -> dict[str, Any]:
    baseline_rows_by_pair: dict[str, dict[str, Any]] = {}
    for row in preferences["rows"]:
        baseline_rows_by_pair.setdefault(row["pair_id"], row)
    baseline: dict[str, float] = {}
    for row in baseline_rows_by_pair.values():
        update_elo(
            baseline,
            row["category_a"],
            row["category_b"],
            float(row["baseline_margin_a_minus_b"]),
        )

    positive_strengths = [
        float(row["raw_strength"])
        for row in preferences["rows"]
        if row["control"] == "real" and float(row["raw_strength"]) > 0.0
    ]
    if not positive_strengths:
        return {"baseline_by_category": baseline, "steered_by_emotion": {}, "causal_shift": {}}
    selected_strength = max(positive_strengths)
    steered: dict[str, dict[str, float]] = {}
    for row in preferences["rows"]:
        if row["control"] != "real" or float(row["raw_strength"]) != selected_strength:
            continue
        emotion = row["target_emotion"]
        ratings = steered.setdefault(emotion, {})
        update_elo(
            ratings,
            row["category_a"],
            row["category_b"],
            float(row["margin_a_minus_b"]),
        )
    shifts = {
        emotion: {
            category: rating - baseline.get(category, 1000.0)
            for category, rating in ratings.items()
        }
        for emotion, ratings in steered.items()
    }
    return {
        "baseline_by_category": baseline,
        "steered_by_emotion": steered,
        "causal_shift": shifts,
        "selected_positive_strength": selected_strength,
    }


def summary_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "all_hard_gates_pass": metrics["all_hard_gates_pass"],
        "selected_layer": metrics["selected_layer"],
        "emotions": metrics["emotions"],
        "zero_steering_max_abs_logit_error": metrics["zero_fidelity"][
            "max_abs_logit_error"
        ],
        "positive_strength_mean_expected_effect": metrics["preferences"][
            "positive_strength_summary"
        ]["mean_expected_effect"],
        "dose_response_mean_spearman": metrics["preferences"]["dose_response"][
            "mean_spearman"
        ],
        "interpretation_caveat": INTERPRETATION_CAVEAT,
    }


def run(config: dict[str, Any]) -> Path:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    prototype1_bundle = Path(config["prototype1"]["run_dir"])
    prototype1_config = json.loads((prototype1_bundle / "config.json").read_text())
    prototype1_metrics = json.loads((prototype1_bundle / "metrics.json").read_text())
    prototype1_environment = json.loads((prototype1_bundle / "environment.json").read_text())
    emotions = list(config.get("emotions") or prototype1_config["data"]["emotions"])
    selected_layer = resolve_selected_layer(config, prototype1_metrics)
    vector_path = resolve_artifact_path(
        prototype1_bundle,
        config["prototype1"].get("emotion_vectors", "emotion_vectors.safetensors"),
        "emotion_vectors.safetensors",
    )
    vectors = load_selected_emotion_vectors(vector_path, emotions, selected_layer)
    strengths = [float(value) for value in config["intervention"]["strengths"]]
    if 0.0 not in strengths:
        raise ValueError("intervention.strengths must include 0.0 for fidelity checks")
    pairs = parse_activity_pairs(config["preference_task"]["pairs"])
    expected_categories = {
        emotion: list(categories)
        for emotion, categories in config["preference_task"]["expected_categories"].items()
    }

    model_config = default_model_config(prototype1_config, config)
    device = choose_device(model_config["device"])
    dtype = choose_dtype(model_config["dtype"], device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        revision=model_config["revision"],
        local_files_only=model_config["local_files_only"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        revision=model_config["revision"],
        dtype=dtype,
        local_files_only=model_config["local_files_only"],
    ).to(device)
    model.eval()
    resolved_model_revision = getattr(model.config, "_commit_hash", None)

    layers = decoder_layers(model)
    layer_index = resolve_layer_index(selected_layer, len(layers))
    layer = layers[layer_index]
    hidden_size = int(next(iter(vectors.values())).numel())
    model_hidden_size = int(getattr(model.config, "hidden_size", hidden_size))
    if hidden_size != model_hidden_size:
        raise ValueError(
            f"Vector hidden size {hidden_size} does not match model hidden size {model_hidden_size}"
        )

    token_ids = choice_token_ids(tokenizer)
    missing = [label for label, ids in token_ids.items() if not ids]
    if missing:
        raise ValueError("Could not resolve next-token ids for choices: " + ", ".join(missing))

    preferences = run_preference_interventions(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        device=device,
        vectors=vectors,
        emotions=emotions,
        pairs=pairs,
        strengths=strengths,
        token_ids=token_ids,
        expected_categories=expected_categories,
        random_seed=int(config["controls"]["random_seed"]),
        scale_by_vector_norm=bool(config["intervention"].get("scale_by_vector_norm", False)),
    )
    zero = zero_fidelity(preferences)
    controls = preferences["control_summary"]
    kl = {
        "preference_kl": [
            {
                "target_emotion": row["target_emotion"],
                "control": row["control"],
                "raw_strength": row["raw_strength"],
                "mean_kl_from_baseline": row["mean_kl_from_baseline"],
            }
            for row in preferences["aggregates"]
        ]
    }
    hard_gates = {
        "required_vector_artifacts_loaded": bool(vectors),
        "zero_steering_reproduces_baseline": zero["max_abs_logit_error"]
        <= float(config["gates"]["zero_steering_max_abs_logit_error"]),
        "random_vector_control_recorded": any(
            row["control"] == "random" for row in preferences["rows"]
        ),
        "wrong_emotion_control_recorded": any(
            row["control"] == "wrong_emotion" for row in preferences["rows"]
        ),
        "kl_diagnostics_recorded": bool(kl["preference_kl"]),
        "pairwise_preference_scores_recorded": bool(preferences["rows"]),
    }
    metrics = {
        "all_hard_gates_pass": all(hard_gates.values()),
        "hard_gates": hard_gates,
        "soft_gates": soft_gates(preferences),
        "selected_layer": layer_index,
        "configured_layer": selected_layer,
        "emotions": emotions,
        "hidden_size": hidden_size,
        "prototype1_run_dir": str(prototype1_bundle),
        "artifacts": {"emotion_vectors": str(vector_path)},
        "vector_norms": {emotion: float(vector.norm()) for emotion, vector in vectors.items()},
        "zero_fidelity": zero,
        "preferences": {
            "positive_strength_summary": preferences["positive_strength_summary"],
            "control_summary": controls,
            "dose_response": preferences["dose_response"],
        },
        "interpretation_caveat": INTERPRETATION_CAVEAT,
    }

    created_at = datetime.now(timezone.utc)
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    resolved = json.loads(json.dumps(config))
    resolved["model"] = {
        **model_config,
        "resolved_device": device,
        "resolved_dtype": str(dtype),
        "resolved_revision": resolved_model_revision,
        "number_of_layers": len(layers),
        "resolved_layer_index": layer_index,
    }
    resolved["prototype1"]["resolved_run_dir"] = str(prototype1_bundle)
    resolved["prototype1"]["resolved_emotion_vectors"] = str(vector_path)
    resolved["prototype1"]["resolved_model_revision"] = prototype1_environment.get(
        "resolved_model_revision"
    )
    config_hash = sha256_json(resolved)
    run_id = make_run_id(
        timestamp=timestamp,
        experiment=resolved["experiment"],
        model_name=model_config["name"],
        seed=seed,
        config_hash=config_hash,
    )
    output = Path(config["output_dir"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    diagnostics = output / "diagnostics"
    diagnostics.mkdir()

    code = git_metadata()
    environment = {
        "created_at": created_at.isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "model_name": model_config["name"],
        "requested_revision": model_config["revision"],
        "resolved_model_revision": resolved_model_revision,
        "prototype1_resolved_model_revision": prototype1_environment.get(
            "resolved_model_revision"
        ),
        "code_git_commit": code["commit"],
        "code_git_dirty": code["dirty"],
        "config_sha256": config_hash,
        "run_id": run_id,
    }
    manifest = build_manifest(
        run_id=run_id,
        created_at=created_at.isoformat(),
        config=resolved,
        resolved_model_revision=resolved_model_revision,
        code=code,
    )
    summary = summary_from_metrics(metrics)
    elo = elo_diagnostic(preferences)
    for filename, value in (
        ("config.json", resolved),
        ("metrics.json", metrics),
        ("environment.json", environment),
        ("manifest.json", manifest),
        ("summary.json", summary),
    ):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n")
    for filename, value in (
        ("preference_scores.json", preferences),
        (
            "controls.json",
            {
                "random_vector": {
                    "recorded": hard_gates["random_vector_control_recorded"],
                    "seed": int(config["controls"]["random_seed"]),
                    "matched_norm": True,
                },
                "wrong_emotion": {
                    emotion: wrong_emotion(emotion, emotions) for emotion in emotions
                },
                "opposite_sign_strengths": [value for value in strengths if value < 0.0],
                "control_summary": controls,
            },
        ),
        ("kl.json", kl),
        ("elo.json", elo),
    ):
        (diagnostics / filename).write_text(json.dumps(value, indent=2) + "\n")

    print(json.dumps({"output": str(output), **summary}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Prototype 5 emotion-steered activity preferences"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/prototype5.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

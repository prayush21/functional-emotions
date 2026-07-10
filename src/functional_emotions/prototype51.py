from __future__ import annotations

import argparse
import json
import math
import platform
import random
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from safetensors.torch import load_file
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .analysis import prototype51_statistics, statistics_params
from .instrumentation import decoder_layers, resolve_layer_index, steer_layer_output
from .prototype0 import choose_device, choose_dtype, encode, next_token_logits
from .prototype4 import (
    apply_steering_scale,
    default_model_config,
    kl_from_logits,
    matched_random_vector,
    resolve_artifact_path,
    resolve_selected_layer,
    spearman,
    steered_next_token_logits,
    wrong_emotion,
)
from .prototype5 import choice_token_ids, finite_mean, mean_choice_logit, update_elo
from .tracking import build_manifest, git_metadata, make_run_id, sha256_json


INTERPRETATION_CAVEAT = (
    "These are compact four-emotion vectors with weak valence/arousal geometry. "
    "Prototype 5.1 tests whether the Prototype 5 null result survives more robust "
    "preference measurement, not whether the model has mature human-like emotion "
    "structure."
)

EXTRA_INTERPRETATION_CAVEAT = (
    "Prototype 5.1 is not a new extraction. It tests whether the original Prototype 5 "
    "activity-preference null was caused by brittle A/B token scoring, option-order "
    "bias, missing context, or layer/strength choice."
)

CHOICE_TOKEN_MARGIN = "choice_token_margin"
OPTION_TEXT_LOGPROB_MARGIN = "option_text_logprob_margin"
LABEL_PLUS_TEXT_LOGPROB_MARGIN = "label_plus_text_logprob_margin"
# Length-normalized diagnostic added after the first registered 0.6B run: mean
# per-token logprob instead of the sum, so unequal option lengths cannot bias
# the margin. Extra scoring mode only; option_text_logprob_margin stays the
# pre-registered primary.
OPTION_TEXT_MEAN_LOGPROB_MARGIN = "option_text_mean_logprob_margin"


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def choice_prompt(context: str, activity_a: str, activity_b: str) -> str:
    return (
        f"{context}\n"
        f"A: {activity_a}\n"
        f"B: {activity_b}\n"
        "The better next action is:"
    )


def option_text_prefix(context: str) -> str:
    return f"{context}\nThe better next action is"


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
            }
        )
    if not pairs:
        raise ValueError("At least one activity pair is required")
    return pairs


def parse_contexts(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    contexts = []
    for index, row in enumerate(rows):
        text = row.get("text")
        if not text:
            raise ValueError("Each context needs text")
        contexts.append(
            {
                "id": str(row.get("id") or f"context-{index + 1}"),
                "family": str(row.get("family") or row.get("id") or f"context-{index + 1}"),
                "text": str(text),
            }
        )
    if not contexts:
        raise ValueError("At least one context is required")
    return contexts


def order_swapped_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    swapped = []
    for pair in pairs:
        swapped.append(
            {
                "pair_id": f"{pair['id']}::original",
                "base_pair_id": pair["id"],
                "order": "original",
                "activity_a": pair["activity_a"],
                "activity_b": pair["activity_b"],
                "category_a": pair["category_a"],
                "category_b": pair["category_b"],
            }
        )
        swapped.append(
            {
                "pair_id": f"{pair['id']}::swapped",
                "base_pair_id": pair["id"],
                "order": "swapped",
                "activity_a": pair["activity_b"],
                "activity_b": pair["activity_a"],
                "category_a": pair["category_b"],
                "category_b": pair["category_a"],
            }
        )
    return swapped


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


def choice_margin(logits: Tensor, token_ids: dict[str, list[int]]) -> Tensor:
    return mean_choice_logit(logits, token_ids["A"]) - mean_choice_logit(
        logits, token_ids["B"]
    )


def continuation_logprob(
    *,
    model: Any,
    layer: Any,
    tokenizer: Any,
    prompt: str,
    continuation: str,
    device: str,
    vector: Tensor | None = None,
    strength: float = 0.0,
    length_normalized: bool = False,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not prompt_ids or not continuation_ids:
        return math.nan
    sequence_ids = prompt_ids + continuation_ids
    sequence = torch.tensor([sequence_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(sequence)
    context = (
        steer_layer_output(layer, vector.to(device), strength)
        if vector is not None and strength != 0.0
        else nullcontext()
    )
    with torch.inference_mode(), context:
        logits = model(input_ids=sequence, attention_mask=attention_mask).logits.float()
    log_probs = logits[:, :-1, :].log_softmax(dim=-1)
    targets = sequence[:, 1:]
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    start = len(prompt_ids) - 1
    end = start + len(continuation_ids)
    total = float(token_log_probs[:, start:end].sum().cpu())
    if length_normalized:
        return total / len(continuation_ids)
    return total


def option_continuations(scoring_mode: str, activity_a: str, activity_b: str) -> tuple[str, str]:
    if scoring_mode in {OPTION_TEXT_LOGPROB_MARGIN, OPTION_TEXT_MEAN_LOGPROB_MARGIN}:
        return f" {activity_a}", f" {activity_b}"
    if scoring_mode == LABEL_PLUS_TEXT_LOGPROB_MARGIN:
        return f" A: {activity_a}", f" B: {activity_b}"
    raise ValueError(f"Unsupported option-text scoring mode: {scoring_mode}")


def option_text_margin(
    *,
    model: Any,
    layer: Any,
    tokenizer: Any,
    prefix: str,
    activity_a: str,
    activity_b: str,
    scoring_mode: str,
    device: str,
    vector: Tensor | None = None,
    strength: float = 0.0,
) -> dict[str, float]:
    continuation_a, continuation_b = option_continuations(
        scoring_mode, activity_a, activity_b
    )
    length_normalized = scoring_mode == OPTION_TEXT_MEAN_LOGPROB_MARGIN
    score_a = continuation_logprob(
        model=model,
        layer=layer,
        tokenizer=tokenizer,
        prompt=prefix,
        continuation=continuation_a,
        device=device,
        vector=vector,
        strength=strength,
        length_normalized=length_normalized,
    )
    score_b = continuation_logprob(
        model=model,
        layer=layer,
        tokenizer=tokenizer,
        prompt=prefix,
        continuation=continuation_b,
        device=device,
        vector=vector,
        strength=strength,
        length_normalized=length_normalized,
    )
    return {
        "score_a": score_a,
        "score_b": score_b,
        "margin_a_minus_b": score_a - score_b,
    }


def load_emotion_vectors_for_layer(
    path: Path, emotions: list[str], layer: int
) -> dict[str, Tensor] | None:
    tensors = load_file(path)
    vectors = {}
    for emotion in emotions:
        key = f"{emotion}/layer_{layer}"
        if key not in tensors:
            return None
        vector = tensors[key].float()
        norm = vector.norm()
        if not torch.isfinite(norm) or float(norm) == 0.0:
            raise ValueError(f"Vector {key!r} is zero or non-finite")
        vectors[emotion] = vector
    return vectors


def configured_vector_layers(config: dict[str, Any], prototype1_metrics: dict[str, Any]) -> list[int]:
    intervention = config["intervention"]
    if intervention.get("layers") is not None:
        return [int(layer) for layer in intervention["layers"]]
    if intervention.get("layer") is not None:
        return [int(resolve_selected_layer(config, prototype1_metrics))]
    selected = int(resolve_selected_layer(config, prototype1_metrics))
    return list(range(selected - 3, selected + 4))


def score_condition(
    *,
    model: Any,
    tokenizer: Any,
    layer: Any,
    device: str,
    scoring_mode: str,
    context: dict[str, str],
    pair: dict[str, Any],
    baseline: dict[str, Any],
    target_emotion: str,
    vector_label: str,
    control: str,
    raw_strength: float,
    applied_strength: float,
    vector: Tensor,
    token_ids: dict[str, list[int]],
    expected_categories: dict[str, list[str]],
    vector_layer: int,
    model_layer: int,
) -> dict[str, Any]:
    if scoring_mode == CHOICE_TOKEN_MARGIN:
        logits = steered_next_token_logits(
            model, layer, baseline["inputs"], vector.to(device), applied_strength
        )
        margin = float(choice_margin(logits, token_ids)[0])
        score_a = math.nan
        score_b = math.nan
    else:
        option_scores = option_text_margin(
            model=model,
            layer=layer,
            tokenizer=tokenizer,
            prefix=baseline["prompt"],
            activity_a=pair["activity_a"],
            activity_b=pair["activity_b"],
            scoring_mode=scoring_mode,
            device=device,
            vector=vector,
            strength=applied_strength,
        )
        logits = steered_next_token_logits(
            model, layer, baseline["inputs"], vector.to(device), applied_strength
        )
        margin = option_scores["margin_a_minus_b"]
        score_a = option_scores["score_a"]
        score_b = option_scores["score_b"]

    baseline_margin = float(baseline["margin_a_minus_b"])
    delta = margin - baseline_margin
    direction = expected_direction(
        target_emotion,
        pair["category_a"],
        pair["category_b"],
        expected_categories,
    )
    expected_effect = float(direction * delta) if direction else math.nan
    max_abs_logit_change = float((logits - baseline["logits"]).abs().max())
    return {
        "vector_layer": vector_layer,
        "model_layer": model_layer,
        "scoring_mode": scoring_mode,
        "context_id": context["id"],
        "context_family": context["family"],
        "context": context["text"],
        "pair_id": pair["pair_id"],
        "base_pair_id": pair["base_pair_id"],
        "order": pair["order"],
        "prompt": baseline["prompt"],
        "activity_a": pair["activity_a"],
        "activity_b": pair["activity_b"],
        "category_a": pair["category_a"],
        "category_b": pair["category_b"],
        "target_emotion": target_emotion,
        "vector_label": vector_label,
        "control": control,
        "raw_strength": raw_strength,
        "applied_strength": applied_strength,
        "baseline_score_a": baseline.get("score_a", math.nan),
        "baseline_score_b": baseline.get("score_b", math.nan),
        "score_a": score_a,
        "score_b": score_b,
        "baseline_margin_a_minus_b": baseline_margin,
        "margin_a_minus_b": margin,
        "margin_delta_from_baseline": delta,
        "expected_direction": direction,
        "expected_effect": expected_effect,
        "matches_hypothesis": direction != 0,
        "max_abs_logit_change": max_abs_logit_change,
        "max_abs_score_change": abs(delta),
        "kl_from_baseline": kl_from_logits(baseline["logits"], logits),
    }


def baseline_for(
    *,
    model: Any,
    tokenizer: Any,
    layer: Any,
    device: str,
    scoring_mode: str,
    context: dict[str, str],
    pair: dict[str, Any],
    token_ids: dict[str, list[int]],
) -> dict[str, Any]:
    if scoring_mode == CHOICE_TOKEN_MARGIN:
        prompt = choice_prompt(context["text"], pair["activity_a"], pair["activity_b"])
        inputs = encode(tokenizer, [prompt], device)
        logits = next_token_logits(model, inputs)
        return {
            "prompt": prompt,
            "inputs": inputs,
            "logits": logits,
            "margin_a_minus_b": float(choice_margin(logits, token_ids)[0]),
        }
    prefix = option_text_prefix(context["text"])
    inputs = encode(tokenizer, [prefix], device)
    logits = next_token_logits(model, inputs)
    scores = option_text_margin(
        model=model,
        layer=layer,
        tokenizer=tokenizer,
        prefix=prefix,
        activity_a=pair["activity_a"],
        activity_b=pair["activity_b"],
        scoring_mode=scoring_mode,
        device=device,
    )
    return {
        "prompt": prefix,
        "inputs": inputs,
        "logits": logits,
        "score_a": scores["score_a"],
        "score_b": scores["score_b"],
        "margin_a_minus_b": scores["margin_a_minus_b"],
    }


def run_layer_assay(
    *,
    model: Any,
    tokenizer: Any,
    layer: Any,
    device: str,
    vectors: dict[str, Tensor],
    emotions: list[str],
    pairs: list[dict[str, Any]],
    contexts: list[dict[str, str]],
    scoring_modes: list[str],
    strengths: list[float],
    token_ids: dict[str, list[int]],
    expected_categories: dict[str, list[str]],
    random_seed: int,
    scale_by_vector_norm: bool,
    vector_layer: int,
    model_layer: int,
) -> dict[str, Any]:
    rows = []
    for context in contexts:
        for pair in pairs:
            for scoring_mode in scoring_modes:
                baseline = baseline_for(
                    model=model,
                    tokenizer=tokenizer,
                    layer=layer,
                    device=device,
                    scoring_mode=scoring_mode,
                    context=context,
                    pair=pair,
                    token_ids=token_ids,
                )
                for target in emotions:
                    controls = {
                        "real": (target, vectors[target]),
                        "random": (
                            target,
                            matched_random_vector(
                                vectors[target],
                                random_seed + vector_layer * 1009 + emotions.index(target),
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
                            rows.append(
                                score_condition(
                                    model=model,
                                    tokenizer=tokenizer,
                                    layer=layer,
                                    device=device,
                                    scoring_mode=scoring_mode,
                                    context=context,
                                    pair=pair,
                                    baseline=baseline,
                                    target_emotion=target,
                                    vector_label=vector_label,
                                    control=control,
                                    raw_strength=raw_strength,
                                    applied_strength=applied,
                                    vector=vector,
                                    token_ids=token_ids,
                                    expected_categories=expected_categories,
                                    vector_layer=vector_layer,
                                    model_layer=model_layer,
                                )
                            )
    return {"rows": rows}


def aggregate_rows(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    kl_max: float | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if kl_max is not None and float(row["kl_from_baseline"]) > kl_max:
            continue
        key = tuple(row[key] for key in group_keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        item = dict(zip(group_keys, key, strict=True))
        matched = [row["expected_effect"] for row in group if row["matches_hypothesis"]]
        nonmatched_abs = [
            abs(row["margin_delta_from_baseline"])
            for row in group
            if not row["matches_hypothesis"]
        ]
        item.update(
            {
                "rows": len(group),
                "mean_margin_delta": float(
                    np.mean([row["margin_delta_from_baseline"] for row in group])
                ),
                "mean_expected_effect": finite_mean(
                    [row["expected_effect"] for row in group]
                ),
                "mean_matching_expected_effect": finite_mean(matched),
                "mean_nonmatching_abs_delta": finite_mean(nonmatched_abs),
                "mean_kl_from_baseline": float(
                    np.mean([row["kl_from_baseline"] for row in group])
                ),
                "max_kl_from_baseline": float(max(row["kl_from_baseline"] for row in group)),
                "max_abs_logit_change": float(
                    max(row["max_abs_logit_change"] for row in group)
                ),
                "max_abs_score_change": float(
                    max(row["max_abs_score_change"] for row in group)
                ),
            }
        )
        output.append(item)
    return output


def positive_strength_summary(
    aggregates: list[dict[str, Any]], primary_scoring_mode: str | None = None
) -> dict[str, Any]:
    rows = [
        row
        for row in aggregates
        if row["control"] == "real"
        and float(row["raw_strength"]) > 0.0
        and (primary_scoring_mode is None or row["scoring_mode"] == primary_scoring_mode)
    ]
    return {
        "mean_expected_effect": finite_mean([row["mean_expected_effect"] for row in rows]),
        "mean_matching_expected_effect": finite_mean(
            [row["mean_matching_expected_effect"] for row in rows]
        ),
        "mean_nonmatching_abs_delta": finite_mean(
            [row["mean_nonmatching_abs_delta"] for row in rows]
        ),
    }


def control_summary(
    aggregates: list[dict[str, Any]], primary_scoring_mode: str | None = None
) -> dict[str, Any]:
    summary = {}
    for control in ("real", "random", "wrong_emotion"):
        rows = [
            row
            for row in aggregates
            if row["control"] == control
            and float(row["raw_strength"]) > 0.0
            and (primary_scoring_mode is None or row["scoring_mode"] == primary_scoring_mode)
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


def dose_response(
    aggregates: list[dict[str, Any]], primary_scoring_mode: str | None = None
) -> dict[str, Any]:
    rows = [
        row
        for row in aggregates
        if row["control"] == "real"
        and float(row["raw_strength"]) > 0.0
        and (primary_scoring_mode is None or row["scoring_mode"] == primary_scoring_mode)
    ]
    by_emotion: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_emotion.setdefault(row["target_emotion"], []).append(row)
    correlations = {}
    for emotion, emotion_rows in by_emotion.items():
        by_strength = aggregate_by_strength(emotion_rows)
        correlations[emotion] = spearman(
            [float(row["raw_strength"]) for row in by_strength],
            [float(row["mean_expected_effect"]) for row in by_strength],
        )
    finite = [value for value in correlations.values() if not math.isnan(value)]
    return {
        "spearman_by_emotion": correlations,
        "mean_spearman": float(np.mean(finite)) if finite else math.nan,
    }


def aggregate_by_strength(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row["raw_strength"]), []).append(row)
    output = []
    for strength, group in sorted(grouped.items()):
        output.append(
            {
                "raw_strength": strength,
                "mean_expected_effect": finite_mean(
                    [row["mean_expected_effect"] for row in group]
                ),
            }
        )
    return output


def order_swap_summary(
    rows: list[dict[str, Any]],
    primary_scoring_mode: str,
    kl_max: float | None = None,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["scoring_mode"] == primary_scoring_mode
        and row["control"] == "real"
        and float(row["raw_strength"]) > 0.0
        and (kl_max is None or float(row["kl_from_baseline"]) <= kl_max)
    ]
    original = [row["expected_effect"] for row in eligible if row["order"] == "original"]
    swapped = [row["expected_effect"] for row in eligible if row["order"] == "swapped"]
    original_mean = finite_mean(original)
    swapped_mean = finite_mean(swapped)
    invariant = finite_mean([row["expected_effect"] for row in eligible])
    same_sign = (
        not math.isnan(original_mean)
        and not math.isnan(swapped_mean)
        and (original_mean == 0.0 or swapped_mean == 0.0 or original_mean * swapped_mean > 0.0)
    )
    paired: dict[tuple[Any, ...], dict[str, float]] = {}
    for row in eligible:
        key = (
            row["vector_layer"],
            row["context_id"],
            row["base_pair_id"],
            row["target_emotion"],
            row["raw_strength"],
        )
        paired.setdefault(key, {})[row["order"]] = row["expected_effect"]
    gaps = [
        abs(pair["original"] - pair["swapped"])
        for pair in paired.values()
        if "original" in pair
        and "swapped" in pair
        and not math.isnan(pair["original"])
        and not math.isnan(pair["swapped"])
    ]
    return {
        "primary_scoring_mode": primary_scoring_mode,
        "original_order_mean_expected_effect": original_mean,
        "swapped_order_mean_expected_effect": swapped_mean,
        "order_invariant_mean_expected_effect": invariant,
        "original_and_swapped_have_same_sign": same_sign,
        "mean_abs_original_swapped_expected_effect_gap": finite_mean(gaps),
        "paired_order_swap_rows": len(gaps),
    }


def zero_fidelity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    zero_rows = [row for row in rows if float(row["raw_strength"]) == 0.0]
    max_logit = max((row["max_abs_logit_change"] for row in zero_rows), default=math.inf)
    max_score = max((row["max_abs_score_change"] for row in zero_rows), default=math.inf)
    return {
        "max_abs_logit_error": float(max_logit),
        "max_abs_logprob_or_margin_error": float(max_score),
        "rows": len(zero_rows),
    }


def kl_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition = aggregate_rows(
        rows, ["vector_layer", "scoring_mode", "control", "raw_strength"]
    )
    return {
        "mean_kl_by_layer_strength_control_scoring_mode": [
            {
                "vector_layer": row["vector_layer"],
                "scoring_mode": row["scoring_mode"],
                "control": row["control"],
                "raw_strength": row["raw_strength"],
                "mean_kl_from_baseline": row["mean_kl_from_baseline"],
                "max_kl_from_baseline": row["max_kl_from_baseline"],
            }
            for row in by_condition
        ],
        "max_kl_from_baseline": float(max((row["kl_from_baseline"] for row in rows), default=math.nan)),
    }


def best_layer(
    layer_effects: list[dict[str, Any]], primary_scoring_mode: str
) -> int | None:
    candidates = [
        row
        for row in layer_effects
        if row["scoring_mode"] == primary_scoring_mode and row["control"] == "real"
    ]
    if not candidates:
        return None
    return int(
        max(
            candidates,
            key=lambda row: (
                -math.inf
                if math.isnan(row["mean_expected_effect"])
                else row["mean_expected_effect"]
            ),
        )["vector_layer"]
    )


def soft_gates(
    *,
    aggregates: list[dict[str, Any]],
    filtered_aggregates: list[dict[str, Any]],
    primary_scoring_mode: str,
    order_summary: dict[str, Any],
) -> dict[str, bool]:
    controls = control_summary(filtered_aggregates, primary_scoring_mode)
    positive = positive_strength_summary(filtered_aggregates, primary_scoring_mode)
    dose = dose_response(filtered_aggregates, primary_scoring_mode)
    real_positive = controls["real"]["mean_expected_effect"]
    random_positive = controls["random"]["mean_expected_effect"]
    wrong_positive = controls["wrong_emotion"]["mean_expected_effect"]
    negative_real = [
        row
        for row in aggregates
        if row["control"] == "real"
        and row["scoring_mode"] == primary_scoring_mode
        and float(row["raw_strength"]) < 0.0
    ]
    positive_real = [
        row
        for row in aggregates
        if row["control"] == "real"
        and row["scoring_mode"] == primary_scoring_mode
        and float(row["raw_strength"]) > 0.0
    ]
    negative_mean = finite_mean([row["mean_expected_effect"] for row in negative_real])
    positive_mean = finite_mean([row["mean_expected_effect"] for row in positive_real])
    happy_rows = [
        row
        for row in filtered_aggregates
        if row["control"] == "real"
        and row["scoring_mode"] == primary_scoring_mode
        and row["target_emotion"] == "happy"
        and float(row["raw_strength"]) > 0.0
    ]
    happy_mean = finite_mean([row["mean_expected_effect"] for row in happy_rows])
    matching_mean = positive["mean_matching_expected_effect"]
    nonmatching_mean = positive["mean_nonmatching_abs_delta"]
    order_bias_gap = order_summary["mean_abs_original_swapped_expected_effect_gap"]
    order_reduces_bias = not math.isnan(order_bias_gap) and order_bias_gap >= 0.0
    return {
        "positive_steering_shifts_expected_preferences_option_text": real_positive > 0.0,
        "order_swapped_aggregation_same_sign_or_reduces_order_bias": (
            order_summary["original_and_swapped_have_same_sign"] or order_reduces_bias
        ),
        "opposite_sign_moves_opposite_direction": negative_mean < positive_mean,
        "intended_effect_exceeds_random_control": real_positive > random_positive,
        "intended_effect_exceeds_wrong_emotion_control": real_positive > wrong_positive,
        "dose_response_spearman_positive": dose["mean_spearman"] > 0.0,
        "matching_pairs_stronger_than_nonmatching_pairs": matching_mean > nonmatching_mean,
        "at_least_one_emotion_shows_positive_robust_effect_under_kl_guardrail": any(
            row["mean_expected_effect"] > 0.0
            for row in filtered_aggregates
            if row["control"] == "real"
            and row["scoring_mode"] == primary_scoring_mode
            and float(row["raw_strength"]) > 0.0
        ),
        "happy_shows_positive_robust_effect_under_kl_guardrail": happy_mean > 0.0,
    }


def elo_diagnostic(rows: list[dict[str, Any]], primary_scoring_mode: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["scoring_mode"] == primary_scoring_mode
        and row["control"] == "real"
        and row["order"] == "original"
    ]
    baseline_rows_by_pair: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in selected:
        key = (row["vector_layer"], row["context_id"], row["base_pair_id"])
        baseline_rows_by_pair.setdefault(key, row)
    baseline: dict[str, float] = {}
    for row in baseline_rows_by_pair.values():
        update_elo(
            baseline,
            row["category_a"],
            row["category_b"],
            float(row["baseline_margin_a_minus_b"]),
        )
    positive_strengths = sorted(
        {
            float(row["raw_strength"])
            for row in selected
            if float(row["raw_strength"]) > 0.0
        }
    )
    if not positive_strengths:
        return {"baseline_by_category": baseline, "steered_by_emotion": {}, "causal_shift": {}}
    selected_strength = max(positive_strengths)
    steered: dict[str, dict[str, float]] = {}
    for row in selected:
        if float(row["raw_strength"]) != selected_strength:
            continue
        ratings = steered.setdefault(row["target_emotion"], {})
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
        "primary_scoring_mode": primary_scoring_mode,
    }


def build_preference_metrics(
    rows: list[dict[str, Any]], primary_scoring_mode: str, kl_max: float
) -> dict[str, Any]:
    aggregate_keys = [
        "vector_layer",
        "scoring_mode",
        "target_emotion",
        "control",
        "raw_strength",
    ]
    aggregates = aggregate_rows(rows, aggregate_keys)
    filtered_aggregates = aggregate_rows(rows, aggregate_keys, kl_max=kl_max)
    per_emotion = aggregate_rows(
        rows, ["scoring_mode", "target_emotion", "control", "raw_strength"], kl_max=kl_max
    )
    per_context = aggregate_rows(
        rows,
        ["vector_layer", "scoring_mode", "context_family", "target_emotion", "control"],
        kl_max=kl_max,
    )
    per_scoring_mode = aggregate_rows(
        rows, ["scoring_mode", "target_emotion", "control", "raw_strength"], kl_max=kl_max
    )
    per_layer = aggregate_rows(
        rows, ["vector_layer", "scoring_mode", "target_emotion", "control"], kl_max=kl_max
    )
    order_summary = order_swap_summary(rows, primary_scoring_mode, kl_max=kl_max)
    return {
        "aggregates": aggregates,
        "kl_filtered_aggregates": filtered_aggregates,
        "positive_strength_summary": positive_strength_summary(
            filtered_aggregates, primary_scoring_mode
        ),
        "control_summary": control_summary(filtered_aggregates, primary_scoring_mode),
        "dose_response": dose_response(filtered_aggregates, primary_scoring_mode),
        "per_emotion": per_emotion,
        "per_context_family": per_context,
        "per_scoring_mode": per_scoring_mode,
        "per_layer": per_layer,
        "order_swap_summary": order_summary,
        "best_diagnostic_layer": best_layer(per_layer, primary_scoring_mode),
    }


def summary_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "all_hard_gates_pass": metrics["all_hard_gates_pass"],
        "configured_layers": metrics["configured_layers"],
        "evaluated_layers": metrics["evaluated_layers"],
        "best_diagnostic_layer": metrics["preferences"]["best_diagnostic_layer"],
        "primary_scoring_mode": metrics["primary_scoring_mode"],
        "emotions": metrics["emotions"],
        "zero_steering_max_abs_logit_error": metrics["zero_fidelity"][
            "max_abs_logit_error"
        ],
        "zero_steering_max_abs_logprob_or_margin_error": metrics["zero_fidelity"][
            "max_abs_logprob_or_margin_error"
        ],
        "positive_strength_mean_expected_effect": metrics["preferences"][
            "positive_strength_summary"
        ]["mean_expected_effect"],
        "dose_response_mean_spearman": metrics["preferences"]["dose_response"][
            "mean_spearman"
        ],
        "kl_max_for_effect_summary": metrics["kl_max_for_effect_summary"],
        "interpretation_caveat": INTERPRETATION_CAVEAT,
        "additional_interpretation_caveat": EXTRA_INTERPRETATION_CAVEAT,
    }
    statistics = metrics.get("statistics")
    if statistics:
        effect = statistics["expected_effect"]["bootstrap_by_pair"]
        summary["expected_effect_row_mean"] = effect["estimate"]
        summary["expected_effect_ci_low"] = effect["ci_low"]
        summary["expected_effect_ci_high"] = effect["ci_high"]
        summary["expected_effect_sign_flip_p_value"] = statistics["expected_effect"][
            "sign_flip_by_pair"
        ]["p_value"]
    return summary


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
    vector_layers = configured_vector_layers(config, prototype1_metrics)
    vector_path = resolve_artifact_path(
        prototype1_bundle,
        config["prototype1"].get("emotion_vectors", "emotion_vectors.safetensors"),
        "emotion_vectors.safetensors",
    )
    strengths = [float(value) for value in config["intervention"]["strengths"]]
    if 0.0 not in strengths:
        raise ValueError("intervention.strengths must include 0.0 for fidelity checks")
    scoring_modes = list(config["scoring"]["modes"])
    primary_scoring_mode = str(config["scoring"].get("primary", OPTION_TEXT_LOGPROB_MARGIN))
    if primary_scoring_mode not in scoring_modes:
        raise ValueError("scoring.primary must be included in scoring.modes")
    if OPTION_TEXT_LOGPROB_MARGIN not in scoring_modes:
        raise ValueError("Prototype 5.1 requires option_text_logprob_margin diagnostics")
    pairs = order_swapped_pairs(parse_activity_pairs(config["preference_task"]["pairs"]))
    contexts = parse_contexts(config["preference_task"]["contexts"])
    expected_categories = {
        emotion: list(categories)
        for emotion, categories in config["preference_task"]["expected_categories"].items()
    }
    kl_max = float(config["gates"].get("kl_max_for_effect_summary", 0.25))

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
    token_ids = choice_token_ids(tokenizer)
    missing = [label for label, ids in token_ids.items() if not ids]
    if missing:
        raise ValueError("Could not resolve next-token ids for choices: " + ", ".join(missing))

    rows = []
    evaluated_layers = []
    skipped_layers = []
    vector_norms = {}
    hidden_size = None
    model_hidden_size = int(getattr(model.config, "hidden_size", 0))
    for vector_layer in vector_layers:
        vectors = load_emotion_vectors_for_layer(vector_path, emotions, vector_layer)
        if vectors is None:
            skipped_layers.append({"vector_layer": vector_layer, "reason": "missing_vectors"})
            continue
        try:
            model_layer_index = resolve_layer_index(vector_layer, len(layers))
        except IndexError:
            skipped_layers.append({"vector_layer": vector_layer, "reason": "missing_model_layer"})
            continue
        layer_hidden_size = int(next(iter(vectors.values())).numel())
        if model_hidden_size and layer_hidden_size != model_hidden_size:
            raise ValueError(
                f"Vector hidden size {layer_hidden_size} does not match model hidden size "
                f"{model_hidden_size}"
            )
        hidden_size = layer_hidden_size
        vector_norms[str(vector_layer)] = {
            emotion: float(vector.norm()) for emotion, vector in vectors.items()
        }
        layer_result = run_layer_assay(
            model=model,
            tokenizer=tokenizer,
            layer=layers[model_layer_index],
            device=device,
            vectors=vectors,
            emotions=emotions,
            pairs=pairs,
            contexts=contexts,
            scoring_modes=scoring_modes,
            strengths=strengths,
            token_ids=token_ids,
            expected_categories=expected_categories,
            random_seed=int(config["controls"]["random_seed"]),
            scale_by_vector_norm=bool(
                config["intervention"].get("scale_by_vector_norm", False)
            ),
            vector_layer=vector_layer,
            model_layer=model_layer_index,
        )
        rows.extend(layer_result["rows"])
        evaluated_layers.append(vector_layer)
    if not evaluated_layers:
        raise ValueError("No configured layer had all required emotion vectors and model layers")

    preferences = build_preference_metrics(rows, primary_scoring_mode, kl_max)
    statistics = prototype51_statistics(
        rows,
        primary_scoring_mode=primary_scoring_mode,
        kl_max=kl_max,
        **statistics_params(config),
    )
    zero = zero_fidelity(rows)
    kl = kl_summary(rows)
    controls = preferences["control_summary"]
    hard_gates = {
        "required_vector_and_model_artifacts_loaded": vector_path.is_file()
        and bool(evaluated_layers),
        "at_least_one_configured_layer_has_all_required_emotion_vectors": bool(
            evaluated_layers
        ),
        "zero_steering_reproduces_baseline_logits_and_logprobs": (
            zero["max_abs_logit_error"]
            <= float(config["gates"]["zero_steering_max_abs_logit_error"])
            and zero["max_abs_logprob_or_margin_error"]
            <= float(config["gates"]["zero_steering_max_abs_logprob_error"])
        ),
        "order_swap_diagnostics_recorded": (
            any(row["order"] == "original" for row in rows)
            and any(row["order"] == "swapped" for row in rows)
        ),
        "option_text_scoring_diagnostics_recorded": any(
            row["scoring_mode"] == OPTION_TEXT_LOGPROB_MARGIN for row in rows
        ),
        "random_vector_control_recorded": any(row["control"] == "random" for row in rows),
        "wrong_emotion_control_recorded": any(
            row["control"] == "wrong_emotion" for row in rows
        ),
        "kl_diagnostics_recorded": bool(
            kl["mean_kl_by_layer_strength_control_scoring_mode"]
        ),
        "pairwise_preference_scores_recorded": bool(rows),
    }
    metrics = {
        "all_hard_gates_pass": all(hard_gates.values()),
        "hard_gates": hard_gates,
        "soft_gates": soft_gates(
            aggregates=preferences["aggregates"],
            filtered_aggregates=preferences["kl_filtered_aggregates"],
            primary_scoring_mode=primary_scoring_mode,
            order_summary=preferences["order_swap_summary"],
        ),
        "configured_layers": vector_layers,
        "evaluated_layers": evaluated_layers,
        "skipped_layers": skipped_layers,
        "primary_scoring_mode": primary_scoring_mode,
        "scoring_modes": scoring_modes,
        "emotions": emotions,
        "hidden_size": hidden_size,
        "prototype1_run_dir": str(prototype1_bundle),
        "artifacts": {"emotion_vectors": str(vector_path)},
        "vector_norms": vector_norms,
        "zero_fidelity": zero,
        "kl_max_for_effect_summary": kl_max,
        "preferences": preferences,
        "statistics": statistics,
        "interpretation_caveat": INTERPRETATION_CAVEAT,
        "additional_interpretation_caveat": EXTRA_INTERPRETATION_CAVEAT,
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
    }
    resolved["intervention"]["resolved_evaluated_layers"] = evaluated_layers
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
    elo = elo_diagnostic(rows, primary_scoring_mode)
    for filename, value in (
        ("config.json", resolved),
        ("metrics.json", metrics),
        ("environment.json", environment),
        ("manifest.json", manifest),
        ("summary.json", summary),
    ):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n")
    for filename, value in (
        (
            "preference_scores.json",
            {
                "rows": rows,
                "aggregates": preferences["aggregates"],
                "kl_filtered_aggregates": preferences["kl_filtered_aggregates"],
            },
        ),
        (
            "order_swap_scores.json",
            {
                "summary": preferences["order_swap_summary"],
                "rows": [row for row in rows if row["scoring_mode"] == primary_scoring_mode],
            },
        ),
        (
            "option_text_scores.json",
            {
                "rows": [
                    row
                    for row in rows
                    if row["scoring_mode"]
                    in {
                        OPTION_TEXT_LOGPROB_MARGIN,
                        OPTION_TEXT_MEAN_LOGPROB_MARGIN,
                        LABEL_PLUS_TEXT_LOGPROB_MARGIN,
                    }
                ]
            },
        ),
        (
            "layer_sweep.json",
            {
                "configured_layers": vector_layers,
                "evaluated_layers": evaluated_layers,
                "skipped_layers": skipped_layers,
                "per_layer": preferences["per_layer"],
                "best_diagnostic_layer": preferences["best_diagnostic_layer"],
            },
        ),
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
        description="Run Prototype 5.1 robust emotion-steered activity preferences"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/prototype51.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

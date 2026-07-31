from __future__ import annotations

import argparse
import json
import math
import platform
import random
import re
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

from .analysis import prototype4_statistics, statistics_params
from .instrumentation import decoder_layers, resolve_layer_index, steer_layer_output
from .prototype0 import choose_device, choose_dtype, encode, next_token_logits
from .tracking import build_manifest, git_metadata, make_run_id, sha256_json


INTERPRETATION_CAVEAT = (
    "These are compact four-emotion vectors with weak valence/arousal geometry. "
    "Prototype 4 tests local causal efficacy and specificity, not mature human-like "
    "emotion structure."
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def resolve_artifact_path(bundle: Path, configured: str, fallback: str) -> Path:
    path = Path(configured)
    if path.is_file():
        return path
    bundled = bundle / fallback
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(f"Could not find {configured!r} or {bundled}")


def load_selected_emotion_vectors(
    path: Path, emotions: list[str], selected_layer: int
) -> dict[str, Tensor]:
    tensors = load_file(path)
    vectors = {}
    for emotion in emotions:
        key = f"{emotion}/layer_{selected_layer}"
        if key not in tensors:
            raise ValueError(f"Missing vector tensor {key!r}")
        vector = tensors[key].float()
        norm = vector.norm()
        if not torch.isfinite(norm) or float(norm) == 0.0:
            raise ValueError(f"Vector {key!r} is zero or non-finite")
        vectors[emotion] = vector
    return vectors


def token_ids_for_terms(tokenizer: Any, terms: list[str]) -> list[int]:
    ids = []
    for term in terms:
        variants = [term, f" {term}"]
        for variant in variants:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                ids.append(int(encoded[0]))
    return sorted(set(ids))


def emotion_token_ids(
    tokenizer: Any, emotions: list[str], terms_by_emotion: dict[str, list[str]]
) -> dict[str, list[int]]:
    return {
        emotion: token_ids_for_terms(tokenizer, terms_by_emotion.get(emotion) or [emotion])
        for emotion in emotions
    }


def matched_random_vector(vector: Tensor, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_vector = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
    random_norm = random_vector.norm()
    if not torch.isfinite(random_norm) or float(random_norm) == 0.0:
        raise ValueError("Random-vector control produced a zero or non-finite vector")
    return random_vector / random_norm * vector.float().norm()


def wrong_emotion(emotion: str, emotions: list[str]) -> str:
    if len(emotions) < 2:
        raise ValueError("Wrong-emotion control requires at least two emotions")
    index = emotions.index(emotion)
    return emotions[(index + 1) % len(emotions)]


def kl_from_logits(baseline: Tensor, candidate: Tensor) -> float:
    base_logp = baseline.log_softmax(dim=-1)
    candidate_logp = candidate.log_softmax(dim=-1)
    base_p = base_logp.exp()
    return float((base_p * (base_logp - candidate_logp)).sum(dim=-1).mean())


def average_ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2
        position = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return math.nan
    if len(set(values_a)) < 2 or len(set(values_b)) < 2:
        return math.nan
    return float(np.corrcoef(average_ranks(values_a), average_ranks(values_b))[0, 1])


def mean_selected_logits(logits: Tensor, token_ids: list[int]) -> Tensor:
    if not token_ids:
        return torch.full((logits.shape[0],), math.nan)
    return logits[:, token_ids].mean(dim=1)


def matching_score_row(
    *,
    prompt_id: str,
    prompt: str,
    target_emotion: str,
    vector_label: str,
    control: str,
    raw_strength: float,
    applied_strength: float,
    baseline_logits: Tensor,
    logits: Tensor,
    token_ids: dict[str, list[int]],
) -> dict[str, Any]:
    baseline_scores = {
        emotion: float(mean_selected_logits(baseline_logits, ids)[0])
        for emotion, ids in token_ids.items()
    }
    steered_scores = {
        emotion: float(mean_selected_logits(logits, ids)[0])
        for emotion, ids in token_ids.items()
    }
    deltas = {
        emotion: steered_scores[emotion] - baseline_scores[emotion]
        for emotion in steered_scores
    }
    non_target = [
        value for emotion, value in deltas.items() if emotion != target_emotion
    ]
    non_target_mean = float(np.mean(non_target)) if non_target else math.nan
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "target_emotion": target_emotion,
        "vector_label": vector_label,
        "control": control,
        "raw_strength": raw_strength,
        "applied_strength": applied_strength,
        "scores": steered_scores,
        "baseline_scores": baseline_scores,
        "deltas": deltas,
        "target_delta": deltas[target_emotion],
        "non_target_mean_delta": non_target_mean,
        "specificity_delta": deltas[target_emotion] - non_target_mean,
        "max_abs_logit_change": float((logits - baseline_logits).abs().max()),
        "kl_from_baseline": kl_from_logits(baseline_logits, logits),
    }


def aggregate_matching_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["target_emotion"], row["control"], float(row["raw_strength"]))
        grouped.setdefault(key, []).append(row)
    output = []
    for (emotion, control, strength), group in sorted(grouped.items()):
        output.append(
            {
                "target_emotion": emotion,
                "control": control,
                "raw_strength": strength,
                "mean_target_delta": float(np.mean([row["target_delta"] for row in group])),
                "mean_specificity_delta": float(
                    np.mean([row["specificity_delta"] for row in group])
                ),
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
        "mean_target_delta": float(np.mean([row["mean_target_delta"] for row in real]))
        if real
        else math.nan,
        "mean_specificity_delta": float(
            np.mean([row["mean_specificity_delta"] for row in real])
        )
        if real
        else math.nan,
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
            "mean_target_delta": float(
                np.mean([row["mean_target_delta"] for row in rows])
            )
            if rows
            else math.nan,
            "mean_specificity_delta": float(
                np.mean([row["mean_specificity_delta"] for row in rows])
            )
            if rows
            else math.nan,
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
            [float(row["mean_target_delta"]) for row in emotion_rows],
        )
    finite = [value for value in correlations.values() if not math.isnan(value)]
    return {
        "spearman_by_emotion": correlations,
        "mean_spearman": float(np.mean(finite)) if finite else math.nan,
    }


def apply_steering_scale(
    raw_strength: float,
    vector: Tensor,
    scale_by_vector_norm: bool,
) -> float:
    if not scale_by_vector_norm:
        return float(raw_strength)
    return float(raw_strength) * float(vector.norm())


def steered_next_token_logits(
    model: Any,
    layer: Any,
    inputs: dict[str, Tensor],
    vector: Tensor,
    strength: float,
) -> Tensor:
    with torch.inference_mode(), steer_layer_output(layer, vector, strength):
        return model(**inputs).logits[:, -1, :].float().cpu()


def generate_with_optional_steering(
    model: Any,
    layer: Any,
    tokenizer: Any,
    inputs: dict[str, Tensor],
    vector: Tensor | None,
    strength: float,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
) -> Tensor:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
    if vector is None or strength == 0.0:
        with torch.inference_mode():
            return model.generate(**inputs, **kwargs)
    with torch.inference_mode(), steer_layer_output(layer, vector, strength):
        return model.generate(**inputs, **kwargs)


def average_continuation_logprob(
    model: Any,
    layer: Any,
    sequence: Tensor,
    prompt_length: int,
    vector: Tensor | None,
    strength: float,
    pad_token_id: int | None,
) -> float:
    attention_mask = torch.ones_like(sequence)
    if pad_token_id is not None:
        attention_mask = (sequence != pad_token_id).long()
    inputs = {"input_ids": sequence, "attention_mask": attention_mask}
    context = (
        steer_layer_output(layer, vector, strength)
        if vector is not None and strength != 0.0
        else _null_context()
    )
    with torch.inference_mode(), context:
        logits = model(**inputs).logits.float()
    if sequence.shape[1] <= prompt_length:
        return math.nan
    log_probs = logits[:, :-1, :].log_softmax(dim=-1)
    targets = sequence[:, 1:]
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    continuation_positions = token_log_probs[:, prompt_length - 1 :]
    return float(continuation_positions.mean())


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


def run_matching_token_interventions(
    *,
    model: Any,
    tokenizer: Any,
    layer: Any,
    device: str,
    vectors: dict[str, Tensor],
    emotions: list[str],
    prompts: list[dict[str, str]],
    strengths: list[float],
    token_ids: dict[str, list[int]],
    random_seed: int,
    scale_by_vector_norm: bool,
) -> dict[str, Any]:
    rows = []
    for prompt_row in prompts:
        prompt = prompt_row["text"]
        prompt_id = prompt_row["id"]
        target = prompt_row["target_emotion"]
        inputs = encode(tokenizer, [prompt], device)
        baseline = next_token_logits(model, inputs)
        controls = {
            "real": (target, vectors[target]),
            "random": (
                target,
                matched_random_vector(vectors[target], random_seed + emotions.index(target)),
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
                    matching_score_row(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        target_emotion=target,
                        vector_label=vector_label,
                        control=control,
                        raw_strength=raw_strength,
                        applied_strength=applied,
                        baseline_logits=baseline,
                        logits=logits,
                        token_ids=token_ids,
                    )
                )
    aggregates = aggregate_matching_rows(rows)["rows"]
    return {
        "token_ids": token_ids,
        "rows": rows,
        "aggregates": aggregates,
        "positive_strength_summary": positive_strength_summary(aggregates),
        "control_summary": control_summary(aggregates),
        "dose_response": dose_response(aggregates),
    }


def run_free_generation_interventions(
    *,
    model: Any,
    tokenizer: Any,
    layer: Any,
    device: str,
    vectors: dict[str, Tensor],
    prompts: list[dict[str, str]],
    strengths: list[float],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    samples_per_condition: int,
    seed: int,
    terms_by_emotion: dict[str, list[str]],
    scale_by_vector_norm: bool,
) -> dict[str, Any]:
    samples = []
    for prompt_row in prompts:
        prompt = prompt_row["text"]
        target = prompt_row["target_emotion"]
        vector = vectors[target]
        for raw_strength in strengths:
            raw_strength = float(raw_strength)
            applied = apply_steering_scale(raw_strength, vector, scale_by_vector_norm)
            for sample_index in range(samples_per_condition):
                torch.manual_seed(seed + len(samples))
                inputs = encode(tokenizer, [prompt], device)
                prompt_length = int(inputs["attention_mask"].sum(dim=1)[0])
                generated = generate_with_optional_steering(
                    model=model,
                    layer=layer,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    vector=vector.to(device),
                    strength=applied,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
                continuation = generated[:, prompt_length:]
                text = tokenizer.batch_decode(continuation.cpu(), skip_special_tokens=True)[0]
                fluency = average_continuation_logprob(
                    model=model,
                    layer=layer,
                    sequence=generated,
                    prompt_length=prompt_length,
                    vector=vector.to(device),
                    strength=applied,
                    pad_token_id=tokenizer.pad_token_id,
                )
                lexical = lexical_emotion_scores(text, terms_by_emotion, target)
                samples.append(
                    {
                        "prompt_id": prompt_row["id"],
                        "prompt": prompt,
                        "target_emotion": target,
                        "sample_index": sample_index,
                        "raw_strength": raw_strength,
                        "applied_strength": applied,
                        "continuation": text,
                        "average_token_logprob": fluency,
                        "perplexity_like": (
                            math.exp(-fluency) if not math.isnan(fluency) else math.nan
                        ),
                        **lexical,
                    }
                )
    return {
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "samples_per_condition": samples_per_condition,
            "seed": seed,
        },
        "lexical_terms": terms_by_emotion,
        "samples": samples,
    }


def lexical_emotion_scores(
    text: str, terms_by_emotion: dict[str, list[str]], target_emotion: str
) -> dict[str, Any]:
    counts = {
        emotion: sum(count_term_occurrences(text, term) for term in terms)
        for emotion, terms in terms_by_emotion.items()
    }
    target_count = counts.get(target_emotion, 0)
    non_target_counts = [
        count for emotion, count in counts.items() if emotion != target_emotion
    ]
    non_target_mean = float(np.mean(non_target_counts)) if non_target_counts else 0.0
    return {
        "emotion_term_counts": counts,
        "target_term_count": target_count,
        "non_target_mean_term_count": non_target_mean,
        "lexical_specificity": float(target_count - non_target_mean),
    }


def count_term_occurrences(text: str, term: str) -> int:
    return len(
        re.findall(
            rf"(?<![A-Za-z]){re.escape(term.lower())}(?![A-Za-z])",
            text.lower(),
        )
    )


def summarize_free_generation(samples: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [
        sample["average_token_logprob"]
        for sample in samples
        if not math.isnan(sample["average_token_logprob"])
    ]
    positive = [
        sample for sample in samples if float(sample.get("raw_strength", 0.0)) > 0.0
    ]
    zero = [
        sample for sample in samples if float(sample.get("raw_strength", 0.0)) == 0.0
    ]
    return {
        "sample_count": len(samples),
        "mean_average_token_logprob": float(np.mean(finite)) if finite else math.nan,
        "minimum_average_token_logprob": min(finite) if finite else math.nan,
        "mean_positive_target_term_count": float(
            np.mean([sample["target_term_count"] for sample in positive])
        )
        if positive
        else math.nan,
        "mean_zero_target_term_count": float(
            np.mean([sample["target_term_count"] for sample in zero])
        )
        if zero
        else math.nan,
        "mean_positive_lexical_specificity": float(
            np.mean([sample["lexical_specificity"] for sample in positive])
        )
        if positive
        else math.nan,
        "mean_zero_lexical_specificity": float(
            np.mean([sample["lexical_specificity"] for sample in zero])
        )
        if zero
        else math.nan,
    }


def zero_fidelity(matching: dict[str, Any]) -> dict[str, Any]:
    zero_rows = [row for row in matching["rows"] if float(row["raw_strength"]) == 0.0]
    max_error = max((row["max_abs_logit_change"] for row in zero_rows), default=math.inf)
    return {"max_abs_logit_error": float(max_error), "rows": len(zero_rows)}


def soft_gates(matching: dict[str, Any]) -> dict[str, bool]:
    aggregates = matching["aggregates"]
    controls = matching["control_summary"]
    dose = matching["dose_response"]
    real_positive = controls["real"]["mean_target_delta"]
    random_positive = controls["random"]["mean_target_delta"]
    wrong_positive = controls["wrong_emotion"]["mean_target_delta"]
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
    negative_mean = (
        float(np.mean([row["mean_target_delta"] for row in negative_real]))
        if negative_real
        else math.nan
    )
    positive_mean = (
        float(np.mean([row["mean_target_delta"] for row in positive_real]))
        if positive_real
        else math.nan
    )
    return {
        "intended_emotion_logit_improves_for_positive_steering": real_positive > 0.0,
        "opposite_sign_moves_opposite_direction": negative_mean < positive_mean,
        "intended_effect_exceeds_random_control": real_positive > random_positive,
        "intended_effect_exceeds_wrong_emotion_control": real_positive > wrong_positive,
        "dose_response_spearman_positive": dose["mean_spearman"] > 0.0,
        "positive_steering_is_specific": controls["real"]["mean_specificity_delta"] > 0.0,
    }


def summary_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "all_hard_gates_pass": metrics["all_hard_gates_pass"],
        "selected_layer": metrics["selected_layer"],
        "emotions": metrics["emotions"],
        "zero_steering_max_abs_logit_error": metrics["zero_fidelity"][
            "max_abs_logit_error"
        ],
        "positive_strength_mean_target_delta": metrics["matching_token"][
            "positive_strength_summary"
        ]["mean_target_delta"],
        "positive_strength_mean_specificity_delta": metrics["matching_token"][
            "positive_strength_summary"
        ]["mean_specificity_delta"],
        "dose_response_mean_spearman": metrics["matching_token"]["dose_response"][
            "mean_spearman"
        ],
        "free_generation_sample_count": metrics["free_generation"]["sample_count"],
        "free_generation_mean_positive_target_term_count": metrics["free_generation"][
            "mean_positive_target_term_count"
        ],
        "free_generation_mean_positive_lexical_specificity": metrics["free_generation"][
            "mean_positive_lexical_specificity"
        ],
        "interpretation_caveat": INTERPRETATION_CAVEAT,
    }
    statistics = metrics.get("statistics")
    if statistics:
        target = statistics["target_delta"]["bootstrap"]
        specificity = statistics["specificity_delta"]["bootstrap"]
        summary["target_delta_ci_low"] = target["ci_low"]
        summary["target_delta_ci_high"] = target["ci_high"]
        summary["specificity_delta_ci_low"] = specificity["ci_low"]
        summary["specificity_delta_ci_high"] = specificity["ci_high"]
        summary["specificity_minus_target_ci_excludes_zero"] = statistics[
            "adjudication"
        ]["ci_excludes_zero"]
    return summary


def default_model_config(prototype1_config: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config.get("model") or {})
    source = prototype1_config.get("model", {})
    local_only = model.get("local_files_only")
    if local_only is None:
        local_only = source.get("local_files_only", False)
    return {
        "name": model.get("name") or source.get("name") or "Qwen/Qwen3-0.6B-Base",
        "revision": model.get("revision") or source.get("revision") or "main",
        "dtype": model.get("dtype") or source.get("dtype") or "auto",
        "device": model.get("device") or source.get("device") or "auto",
        "local_files_only": bool(local_only),
    }


def resolve_selected_layer(config: dict[str, Any], prototype1_metrics: dict[str, Any]) -> int:
    configured = config["intervention"].get("layer")
    if configured is not None:
        return int(configured)
    if prototype1_metrics.get("selected_layer") is not None:
        return int(prototype1_metrics["selected_layer"])
    return 19


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

    token_ids = emotion_token_ids(
        tokenizer, emotions, config["matching_token"].get("terms", {})
    )
    missing_tokens = [emotion for emotion, ids in token_ids.items() if not ids]
    if missing_tokens:
        raise ValueError(
            "Could not resolve next-token ids for emotions: " + ", ".join(missing_tokens)
        )

    matching = run_matching_token_interventions(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        device=device,
        vectors=vectors,
        emotions=emotions,
        prompts=config["matching_token"]["prompts"],
        strengths=strengths,
        token_ids=token_ids,
        random_seed=int(config["controls"]["random_seed"]),
        scale_by_vector_norm=bool(config["intervention"].get("scale_by_vector_norm", False)),
    )
    free_generation_config = config["free_generation"]
    free_generation_strengths = [
        float(value) for value in free_generation_config.get("strengths", strengths)
    ]
    free_generation_terms = (
        free_generation_config.get("terms")
        or config["matching_token"].get("terms", {})
    )
    free_generation = run_free_generation_interventions(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        device=device,
        vectors=vectors,
        prompts=free_generation_config["prompts"],
        strengths=free_generation_strengths,
        max_new_tokens=int(free_generation_config["max_new_tokens"]),
        do_sample=bool(free_generation_config.get("do_sample", False)),
        temperature=(
            float(free_generation_config["temperature"])
            if free_generation_config.get("temperature") is not None
            else None
        ),
        top_p=(
            float(free_generation_config["top_p"])
            if free_generation_config.get("top_p") is not None
            else None
        ),
        samples_per_condition=int(free_generation_config.get("samples_per_condition", 1)),
        seed=int(free_generation_config.get("seed", seed)),
        terms_by_emotion=free_generation_terms,
        scale_by_vector_norm=bool(config["intervention"].get("scale_by_vector_norm", False)),
    )
    free_generation_summary = summarize_free_generation(free_generation["samples"])
    zero = zero_fidelity(matching)
    controls = matching["control_summary"]
    kl_fluency = {
        "matching_token_kl": [
            {
                "target_emotion": row["target_emotion"],
                "control": row["control"],
                "raw_strength": row["raw_strength"],
                "mean_kl_from_baseline": row["mean_kl_from_baseline"],
            }
            for row in matching["aggregates"]
        ],
        "free_generation_fluency": free_generation["samples"],
        "free_generation_summary": free_generation_summary,
    }
    hard_gates = {
        "required_vector_artifacts_loaded": bool(vectors),
        "zero_steering_reproduces_baseline": zero["max_abs_logit_error"]
        <= float(config["gates"]["zero_steering_max_abs_logit_error"]),
        "random_vector_control_recorded": any(
            row["control"] == "random" for row in matching["rows"]
        ),
        "kl_diagnostics_recorded": bool(kl_fluency["matching_token_kl"]),
        "fluency_diagnostics_recorded": bool(kl_fluency["free_generation_fluency"]),
    }
    metrics = {
        "all_hard_gates_pass": all(hard_gates.values()),
        "hard_gates": hard_gates,
        "soft_gates": soft_gates(matching),
        "selected_layer": layer_index,
        "configured_layer": selected_layer,
        "emotions": emotions,
        "hidden_size": hidden_size,
        "prototype1_run_dir": str(prototype1_bundle),
        "artifacts": {"emotion_vectors": str(vector_path)},
        "vector_norms": {emotion: float(vector.norm()) for emotion, vector in vectors.items()},
        "zero_fidelity": zero,
        "matching_token": {
            "positive_strength_summary": matching["positive_strength_summary"],
            "control_summary": controls,
            "dose_response": matching["dose_response"],
        },
        "statistics": prototype4_statistics(matching["rows"], **statistics_params(config)),
        "free_generation": free_generation_summary,
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
    for filename, value in (
        ("config.json", resolved),
        ("metrics.json", metrics),
        ("environment.json", environment),
        ("manifest.json", manifest),
        ("summary.json", summary),
    ):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n")
    for filename, value in (
        ("matching_token_scores.json", matching),
        ("free_generation_samples.json", free_generation),
        ("kl_fluency.json", kl_fluency),
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
    ):
        (diagnostics / filename).write_text(json.dumps(value, indent=2) + "\n")

    print(json.dumps({"output": str(output), **summary}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prototype 4 causal emotion steering")
    parser.add_argument("--config", type=Path, default=Path("configs/prototype4.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

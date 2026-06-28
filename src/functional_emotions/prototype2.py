from __future__ import annotations

import argparse
import json
import math
import platform
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from safetensors.torch import load_file
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .instrumentation import decoder_layers
from .prototype0 import choose_device, choose_dtype
from .prototype1 import (
    dataset_sha256,
    difference_in_means,
    evaluate_vectors,
    pooled_activations,
    pooling_diagnostics,
    read_jsonl,
    resolve_evaluation_layer,
    resolve_layer_indices,
    split_topics,
    validate_story_rows,
    write_jsonl,
)
from .tracking import build_manifest, git_metadata, make_run_id, sha256_json


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


def load_emotion_vectors(path: Path, emotions: list[str]) -> tuple[Tensor, list[int]]:
    tensors = load_file(path)
    layers = sorted({int(name.rsplit("_", 1)[1]) for name in tensors})
    vectors = []
    for emotion in emotions:
        emotion_vectors = []
        for layer in layers:
            key = f"{emotion}/layer_{layer}"
            if key not in tensors:
                raise ValueError(f"Missing vector tensor {key!r}")
            emotion_vectors.append(tensors[key].float())
        vectors.append(torch.stack(emotion_vectors))
    return torch.stack(vectors), layers


def safetensor_metadata(path: Path) -> dict[str, Any]:
    tensors = load_file(path)
    return {
        "path": str(path),
        "tensor_count": len(tensors),
        "keys": sorted(tensors)[:20],
    }


def evaluate_score_matrix(scores: Tensor, labels: list[str], emotions: list[str]) -> dict[str, Any]:
    scores = scores.detach().cpu()
    label_ids = torch.tensor([emotions.index(label) for label in labels])
    predicted = scores.argmax(dim=1)
    correct_scores = scores[torch.arange(len(scores)), label_ids]
    masked = scores.clone()
    masked[torch.arange(len(scores)), label_ids] = -torch.inf
    margins = correct_scores - masked.max(dim=1).values
    confusion = [
        [
            int(((label_ids == true_index) & (predicted == predicted_index)).sum())
            for predicted_index in range(len(emotions))
        ]
        for true_index in range(len(emotions))
    ]
    aucs = [
        _binary_auc(scores[:, index], label_ids == index) for index in range(len(emotions))
    ]
    return {
        "accuracy": float((predicted == label_ids).float().mean()),
        "macro_auc": float(np.nanmean(aucs)),
        "mean_correct_margin": float(margins.mean()),
        "confusion_matrix": {
            "labels": emotions,
            "rows_are_true_labels": confusion,
        },
        "per_emotion": {
            emotion: {
                "accuracy": float((predicted[label_ids == index] == index).float().mean()),
                "auc": aucs[index],
            }
            for index, emotion in enumerate(emotions)
        },
    }


def _average_ranks(values: Tensor) -> Tensor:
    order = torch.argsort(values)
    ranks = torch.empty(len(values), dtype=torch.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2
        position = end
    return ranks


def _binary_auc(scores: Tensor, positives: Tensor) -> float:
    positive_count = int(positives.sum())
    negative_count = len(positives) - positive_count
    if not positive_count or not negative_count:
        return math.nan
    ranks = _average_ranks(scores.double())
    positive_rank_sum = float(ranks[positives].sum())
    return (positive_rank_sum - positive_count * (positive_count - 1) / 2) / (
        positive_count * negative_count
    )


def score_layers(activations: Tensor, vectors: Tensor) -> list[Tensor]:
    return [
        activations[:, layer_position] @ vectors[:, layer_position].T
        for layer_position in range(vectors.shape[1])
    ]


def shuffled_label_control(
    activations: Tensor,
    labels: list[str],
    emotions: list[str],
    vectors: Tensor,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    permutation = list(range(len(emotions)))
    rng.shuffle(permutation)
    if len(permutation) > 1 and permutation == list(range(len(emotions))):
        permutation = permutation[1:] + permutation[:1]
    shuffled = vectors[permutation]
    real = evaluate_vectors(activations, labels, emotions, vectors)
    control = evaluate_vectors(activations, labels, emotions, shuffled)
    return [
        {
            "layer_position": index,
            "real": real_row,
            "shuffled": shuffled_row,
            "effect": metric_deltas(real_row, shuffled_row),
            "vector_label_permutation": {
                emotions[target]: emotions[source] for target, source in enumerate(permutation)
            },
        }
        for index, (real_row, shuffled_row) in enumerate(zip(real, control, strict=True))
    ]


def metric_deltas(real: dict[str, Any], control: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(real["accuracy"] - control["accuracy"]),
        "macro_auc": float(real["macro_auc"] - control["macro_auc"]),
        "mean_correct_margin": float(
            real["mean_correct_margin"] - control["mean_correct_margin"]
        ),
    }


def pca_comparison(
    raw_validation: list[dict[str, Any]],
    clean_validation: list[dict[str, Any]],
    layers: list[int],
) -> list[dict[str, Any]]:
    return [
        {
            "layer": layer,
            "raw": raw,
            "pca_cleaned": clean,
            "pca_minus_raw": metric_deltas(clean, raw),
        }
        for layer, raw, clean in zip(layers, raw_validation, clean_validation, strict=True)
    ]


def best_layers(rows: list[dict[str, Any]], vector_key: str) -> dict[str, dict[str, Any]]:
    output = {}
    for metric in ("accuracy", "macro_auc", "mean_correct_margin"):
        best = max(rows, key=lambda row: row[vector_key][metric])
        output[metric] = {
            "layer": best["layer"],
            "value": best[vector_key][metric],
            "diagnostic_only": True,
        }
    return output


def confusion_diagnostics(
    scores_by_layer: list[Tensor],
    validation: list[dict[str, Any]],
    emotions: list[str],
    layers: list[int],
) -> dict[str, Any]:
    per_layer = []
    afraid_wins_anywhere = False
    angry_dominates_layers = []
    for layer, scores, row in zip(layers, scores_by_layer, validation, strict=True):
        predicted = scores.argmax(dim=1)
        win_counts = Counter(emotions[int(index)] for index in predicted)
        win_rates = {
            emotion: win_counts.get(emotion, 0) / len(predicted) if len(predicted) else 0.0
            for emotion in emotions
        }
        if win_counts.get("afraid", 0) > 0:
            afraid_wins_anywhere = True
        if "angry" in emotions and win_rates["angry"] == max(win_rates.values()):
            angry_dominates_layers.append(layer)
        per_layer.append(
            {
                "layer": layer,
                "win_rates": win_rates,
                "mean_scores": {
                    emotion: float(scores[:, index].mean())
                    for index, emotion in enumerate(emotions)
                },
                "confusion_matrix": row["confusion_matrix"],
            }
        )
    return {
        "afraid_ever_wins_argmax": afraid_wins_anywhere,
        "angry_is_top_win_rate_layers": angry_dominates_layers,
        "angry_top_win_rate_fraction": (
            len(angry_dominates_layers) / len(layers) if layers else 0.0
        ),
        "layers": per_layer,
    }


def calibration_parameters(train_scores: Tensor) -> dict[str, Tensor]:
    means = train_scores.mean(dim=0)
    stds = train_scores.std(dim=0).clamp_min(1e-6)
    return {"means": means, "stds": stds}


def apply_zscore_calibration(scores: Tensor, parameters: dict[str, Tensor]) -> Tensor:
    return (scores - parameters["means"]) / parameters["stds"]


def calibration_diagnostic(
    train_activations: Tensor,
    test_activations: Tensor,
    train_labels: list[str],
    test_labels: list[str],
    emotions: list[str],
    vectors: Tensor,
    layers: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for layer_position, layer in enumerate(layers):
        train_scores = train_activations[:, layer_position] @ vectors[:, layer_position].T
        test_scores = test_activations[:, layer_position] @ vectors[:, layer_position].T
        parameters = calibration_parameters(train_scores)
        calibrated = apply_zscore_calibration(test_scores, parameters)
        uncalibrated_metrics = evaluate_score_matrix(test_scores, test_labels, emotions)
        calibrated_metrics = evaluate_score_matrix(calibrated, test_labels, emotions)
        train_metrics = evaluate_score_matrix(
            apply_zscore_calibration(train_scores, parameters), train_labels, emotions
        )
        rows.append(
            {
                "layer": layer,
                "method": "per-emotion train-score z-score",
                "diagnostic_only": True,
                "train_calibrated": train_metrics,
                "held_out_uncalibrated": uncalibrated_metrics,
                "held_out_calibrated": calibrated_metrics,
                "calibrated_minus_uncalibrated": metric_deltas(
                    calibrated_metrics, uncalibrated_metrics
                ),
                "score_means": {
                    emotion: float(parameters["means"][index])
                    for index, emotion in enumerate(emotions)
                },
                "score_stds": {
                    emotion: float(parameters["stds"][index])
                    for index, emotion in enumerate(emotions)
                },
            }
        )
    return rows


def final_norm_module(model: nn.Module) -> nn.Module | None:
    candidates = (
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
    )
    for path in candidates:
        current: Any = model
        for attribute in path:
            current = getattr(current, attribute, None)
            if current is None:
                break
        if isinstance(current, nn.Module):
            return current
    return None


def emotion_logit_token_ids(
    tokenizer: Any,
    emotions: list[str],
    terms_by_emotion: dict[str, list[str]] | None = None,
) -> dict[str, list[int]]:
    terms_by_emotion = terms_by_emotion or {}
    token_ids = {}
    for emotion in emotions:
        terms = terms_by_emotion.get(emotion) or [emotion]
        ids = []
        for term in terms:
            variants = [term, f" {term}"]
            for variant in variants:
                encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
                if encoded:
                    ids.append(int(encoded[0]))
        token_ids[emotion] = sorted(set(ids))
    return token_ids


def logit_lens_diagnostic(
    model: nn.Module,
    tokenizer: Any,
    activations: Tensor,
    labels: list[str],
    emotions: list[str],
    layers: list[int],
    terms_by_emotion: dict[str, list[str]] | None = None,
    batch_size: int = 4,
) -> dict[str, Any]:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        return {"available": False, "reason": "model_has_no_output_embeddings", "layers": []}
    norm = final_norm_module(model)
    token_ids = emotion_logit_token_ids(tokenizer, emotions, terms_by_emotion)
    if any(not ids for ids in token_ids.values()):
        return {
            "available": False,
            "reason": "missing_emotion_token_ids",
            "emotion_token_ids": token_ids,
            "layers": [],
        }

    device = next(model.parameters()).device
    per_layer = []
    for layer_position, layer in enumerate(layers):
        scores = []
        for offset in range(0, activations.shape[0], batch_size):
            hidden = activations[offset : offset + batch_size, layer_position].to(device)
            hidden = hidden.to(dtype=next(output_embeddings.parameters()).dtype)
            with torch.inference_mode():
                if norm is not None:
                    hidden = norm(hidden)
                logits = output_embeddings(hidden).float().cpu()
            scores.append(
                torch.stack(
                    [
                        logits[:, token_ids[emotion]].mean(dim=1)
                        for emotion in emotions
                    ],
                    dim=1,
                )
            )
        layer_scores = torch.cat(scores) if scores else torch.empty(0, len(emotions))
        per_layer.append(
            {
                "layer": layer,
                "metrics": evaluate_score_matrix(layer_scores, labels, emotions),
                "mean_logits": {
                    emotion: float(layer_scores[:, index].mean())
                    for index, emotion in enumerate(emotions)
                },
                "diagnostic_only": True,
            }
        )
    return {
        "available": True,
        "method": "pooled residual through final norm and output embeddings",
        "emotion_token_ids": token_ids,
        "layers": per_layer,
        "diagnostic_only": True,
    }


def topic_stratified_metrics(
    rows: list[dict[str, Any]],
    scores_by_layer: list[Tensor],
    emotions: list[str],
    layers: list[int],
) -> dict[str, Any]:
    topics = sorted({row["topic"] for row in rows})
    per_layer = []
    for layer, scores in zip(layers, scores_by_layer, strict=True):
        topic_rows = []
        for topic in topics:
            indices = [index for index, row in enumerate(rows) if row["topic"] == topic]
            labels = [rows[index]["emotion"] for index in indices]
            topic_scores = scores[indices]
            topic_rows.append(
                {
                    "topic": topic,
                    "rows": len(indices),
                    "metrics": evaluate_score_matrix(topic_scores, labels, emotions),
                }
            )
        accuracies = [row["metrics"]["accuracy"] for row in topic_rows]
        margins = [row["metrics"]["mean_correct_margin"] for row in topic_rows]
        per_layer.append(
            {
                "layer": layer,
                "topics": topic_rows,
                "minimum_topic_accuracy": min(accuracies) if accuracies else math.nan,
                "mean_topic_accuracy": float(np.mean(accuracies)) if accuracies else math.nan,
                "minimum_topic_margin": min(margins) if margins else math.nan,
                "diagnostic_only": True,
            }
        )
    return {"layers": per_layer, "diagnostic_only": True}


def default_lexical_scenarios() -> list[dict[str, str]]:
    return [
        {
            "id": "lex-happy-1",
            "emotion": "happy",
            "text": "Mira read the final line twice, pressed the paper flat, and began setting an extra place at dinner.",
        },
        {
            "id": "lex-sad-1",
            "emotion": "sad",
            "text": "Jon left the clean mug untouched, folded the note along the same crease, and sat until the room went dark.",
        },
        {
            "id": "lex-sad-control",
            "emotion": "sad",
            "variant": "minimal_control",
            "text": "Jon left the clean mug untouched, folded the note along the same crease, and waited until the room went dark.",
        },
        {
            "id": "lex-angry-1",
            "emotion": "angry",
            "text": "Leah stacked the forms into a hard-edged pile, answered in clipped words, and shut the drawer with both hands.",
        },
        {
            "id": "lex-angry-control",
            "emotion": "angry",
            "variant": "minimal_control",
            "text": "Leah stacked the forms into a neat pile, answered in brief words, and closed the drawer with both hands.",
        },
        {
            "id": "lex-afraid-1",
            "emotion": "afraid",
            "text": "Noor counted the hallway doors, kept one hand on the rail, and stopped speaking whenever the elevator moved.",
        },
        {
            "id": "lex-happy-control",
            "emotion": "happy",
            "variant": "minimal_control",
            "text": "Mira read the final line twice, pressed the paper flat, and began setting an extra place on the table.",
        },
        {
            "id": "lex-afraid-control",
            "emotion": "afraid",
            "variant": "minimal_control",
            "text": "Noor counted the hallway doors, kept one hand near the rail, and paused whenever the elevator moved.",
        },
    ]


def default_intensity_scenarios(emotions: list[str]) -> list[dict[str, Any]]:
    templates = {
        "happy": [
            "After the call, Avery tidied the desk with a lighter rhythm and left the window open.",
            "After the call, Avery crossed the room twice, started three small helpful tasks, and hummed while packing.",
            "After the call, Avery ran upstairs, gathered everyone in the kitchen, and could barely stand still.",
        ],
        "sad": [
            "After reading the note, Rowan moved more slowly and let the kettle cool before pouring.",
            "After reading the note, Rowan skipped the meal, held the sleeve of an old coat, and stopped answering messages.",
            "After reading the note, Rowan sat on the hallway floor, surrounded by unopened boxes, unable to begin again.",
        ],
        "angry": [
            "After the decision, Imani set the pen down carefully and answered with shorter sentences.",
            "After the decision, Imani rewrote the agenda in block letters and pushed each page into a precise stack.",
            "After the decision, Imani cleared the table in one sweep and told everyone to start over from the first line.",
        ],
        "afraid": [
            "After the sound, Ellis checked the window latch once and kept the lamp on.",
            "After the sound, Ellis checked every room, listened at the door, and kept shoes beside the bed.",
            "After the sound, Ellis crouched behind the sofa, phone in hand, waiting for any sign that it had stopped.",
        ],
    }
    rows = []
    for emotion in emotions:
        for level, text in enumerate(templates.get(emotion, []), 1):
            rows.append(
                {
                    "id": f"intensity-{emotion}-{level}",
                    "emotion": emotion,
                    "level": level,
                    "text": text,
                }
            )
    return rows


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    x_ranks = _average_ranks(torch.tensor(xs, dtype=torch.float64))
    y_ranks = _average_ranks(torch.tensor(ys, dtype=torch.float64))
    x_centered = x_ranks - x_ranks.mean()
    y_centered = y_ranks - y_ranks.mean()
    denominator = x_centered.norm() * y_centered.norm()
    if float(denominator) == 0.0:
        return math.nan
    return float((x_centered @ y_centered) / denominator)


def lexical_diagnostic(
    rows: list[dict[str, Any]],
    activations: Tensor,
    emotions: list[str],
    vectors: Tensor,
    layers: list[int],
    forbidden_terms: dict[str, list[str]],
) -> dict[str, Any]:
    audit_rows = [
        {"topic": row.get("id", "lexical"), "emotion": row["emotion"], "text": row["text"]}
        for row in rows
    ]
    audit = validate_story_rows(audit_rows, emotions, forbidden_terms)
    labels = [row["emotion"] for row in rows]
    per_layer = []
    for layer_position, layer in enumerate(layers):
        scores = activations[:, layer_position] @ vectors[:, layer_position].T
        metrics = evaluate_score_matrix(scores, labels, emotions)
        per_layer.append(
            {
                "layer": layer,
                "metrics": metrics,
                "examples": score_examples(rows, scores, emotions),
            }
        )
    return {
        "chance_accuracy": 1 / len(emotions),
        "lexical_leakage": audit["lexical_leakage"],
        "layers": per_layer,
    }


def intensity_diagnostic(
    rows: list[dict[str, Any]],
    activations: Tensor,
    emotions: list[str],
    vectors: Tensor,
    layers: list[int],
    forbidden_terms: dict[str, list[str]],
) -> dict[str, Any]:
    audit_rows = [
        {"topic": row.get("id", "intensity"), "emotion": row["emotion"], "text": row["text"]}
        for row in rows
    ]
    audit = validate_story_rows(audit_rows, emotions, forbidden_terms)
    per_layer = []
    for layer_position, layer in enumerate(layers):
        scores = activations[:, layer_position] @ vectors[:, layer_position].T
        correlations = {}
        for emotion_index, emotion in enumerate(emotions):
            subset = [index for index, row in enumerate(rows) if row["emotion"] == emotion]
            levels = [float(rows[index]["level"]) for index in subset]
            own_scores = [float(scores[index, emotion_index]) for index in subset]
            correlations[emotion] = spearman(levels, own_scores)
        finite = [value for value in correlations.values() if not math.isnan(value)]
        per_layer.append(
            {
                "layer": layer,
                "spearman_by_emotion": correlations,
                "mean_spearman": float(np.mean(finite)) if finite else math.nan,
                "examples": score_examples(rows, scores, emotions),
            }
        )
    return {
        "lexical_leakage": audit["lexical_leakage"],
        "layers": per_layer,
    }


def score_examples(rows: list[dict[str, Any]], scores: Tensor, emotions: list[str]) -> list[dict[str, Any]]:
    examples = []
    for index, row in enumerate(rows):
        predicted = int(scores[index].argmax())
        examples.append(
            {
                "id": row.get("id"),
                "emotion": row["emotion"],
                "level": row.get("level"),
                "variant": row.get("variant"),
                "prediction": emotions[predicted],
                "intended_score": float(scores[index, emotions.index(row["emotion"])]),
                "scores": {
                    emotion: float(scores[index, emotion_index])
                    for emotion_index, emotion in enumerate(emotions)
                },
                "text": row["text"],
            }
        )
    return examples


def selected_layer_row(rows: list[dict[str, Any]], selected_layer: int) -> dict[str, Any]:
    for row in rows:
        if row["layer"] == selected_layer:
            return row
    raise ValueError(f"Layer {selected_layer} was not evaluated")


def answer_summary(metrics: dict[str, Any], selected_layer: int) -> dict[str, Any]:
    selected_pca = selected_layer_row(metrics["pca_comparison"], selected_layer)
    selected_shuffle = selected_layer_row(metrics["shuffled_label_control"], selected_layer)
    selected_calibration = selected_layer_row(metrics["calibration"], selected_layer)
    selected_lexical = selected_layer_row(metrics["lexical_robustness"]["layers"], selected_layer)
    selected_intensity = selected_layer_row(metrics["intensity_sweep"]["layers"], selected_layer)
    selected_topic = selected_layer_row(
        metrics["cross_topic_generalization"]["pca_cleaned"]["layers"], selected_layer
    )
    confusion = metrics["emotion_confusion"]
    selected_logit_lens = (
        selected_layer_row(metrics["logit_lens"]["layers"], selected_layer)
        if metrics["logit_lens"]["available"]
        else None
    )
    return {
        "is_there_signal_beyond_shuffled_labels": (
            selected_shuffle["effect"]["macro_auc"] > 0
            or selected_shuffle["effect"]["accuracy"] > 0
        ),
        "shuffled_label_effect_at_selected_layer": selected_shuffle["effect"],
        "is_signal_stable_across_layers": layer_stability(metrics["pca_comparison"]),
        "cross_topic_generalization": {
            "selected_layer_minimum_topic_accuracy": selected_topic[
                "minimum_topic_accuracy"
            ],
            "selected_layer_mean_topic_accuracy": selected_topic["mean_topic_accuracy"],
            "selected_layer_minimum_topic_margin": selected_topic["minimum_topic_margin"],
        },
        "is_pca_cleaning_helping_at_selected_layer": selected_pca["pca_minus_raw"],
        "logit_lens": {
            "available": metrics["logit_lens"]["available"],
            "selected_layer_accuracy": (
                selected_logit_lens["metrics"]["accuracy"]
                if selected_logit_lens is not None
                else None
            ),
            "selected_layer_macro_auc": (
                selected_logit_lens["metrics"]["macro_auc"]
                if selected_logit_lens is not None
                else None
            ),
            "diagnostic_only": True,
        },
        "afraid_status": {
            "ever_wins_argmax_across_layers": confusion["afraid_ever_wins_argmax"],
            "selected_layer_win_rate": selected_confusion_win_rate(
                confusion, selected_layer, "afraid"
            ),
        },
        "angry_status": {
            "top_win_rate_fraction_across_layers": confusion["angry_top_win_rate_fraction"],
            "selected_layer_win_rate": selected_confusion_win_rate(
                confusion, selected_layer, "angry"
            ),
        },
        "implicit_intensity_monotonicity": {
            "selected_layer_mean_spearman": selected_intensity["mean_spearman"],
            "positive": selected_intensity["mean_spearman"] > 0,
        },
        "lexical_robustness": {
            "selected_layer_accuracy": selected_lexical["metrics"]["accuracy"],
            "beats_chance": selected_lexical["metrics"]["accuracy"]
            > metrics["lexical_robustness"]["chance_accuracy"],
        },
        "calibration_interpretation": {
            "diagnostic_only": True,
            "calibrated_minus_uncalibrated": selected_calibration[
                "calibrated_minus_uncalibrated"
            ],
            "margin_failure_looks_like_score_scale_mismatch": selected_calibration[
                "held_out_calibrated"
            ]["mean_correct_margin"]
            > selected_calibration["held_out_uncalibrated"]["mean_correct_margin"],
        },
        "prototype3_recommendation": prototype3_recommendation(
            selected_shuffle, selected_intensity, selected_lexical
        ),
    }


def layer_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    auc_positive = sum(1 for row in rows if row["pca_cleaned"]["macro_auc"] > 0.5)
    margin_positive = sum(1 for row in rows if row["pca_cleaned"]["mean_correct_margin"] > 0)
    return {
        "layers_above_chance_macro_auc": auc_positive,
        "layers_with_positive_margin": margin_positive,
        "total_layers": len(rows),
    }


def selected_confusion_win_rate(
    confusion: dict[str, Any], selected_layer: int, emotion: str
) -> float | None:
    for row in confusion["layers"]:
        if row["layer"] == selected_layer:
            return row["win_rates"].get(emotion)
    return None


def prototype3_recommendation(
    selected_shuffle: dict[str, Any],
    selected_intensity: dict[str, Any],
    selected_lexical: dict[str, Any],
) -> str:
    lexical_chance = selected_lexical.get("chance_accuracy", 0.0)
    if (
        selected_shuffle["effect"]["macro_auc"] > 0
        and selected_intensity["mean_spearman"] > 0
        and selected_lexical["metrics"]["accuracy"] > lexical_chance
    ):
        return "Proceed to Prototype 3 geometry, while preserving calibration caveats."
    return "Revise extraction/data before treating geometry as substantively meaningful."


def run(config: dict[str, Any]) -> Path:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    prototype1_bundle = Path(config["prototype1"]["run_dir"])
    prototype1_config = json.loads((prototype1_bundle / "config.json").read_text())
    prototype1_metrics = json.loads((prototype1_bundle / "metrics.json").read_text())

    emotions = list(prototype1_config["data"]["emotions"])
    forbidden_terms = prototype1_config["data"].get("forbidden_terms", {})
    stories_path = resolve_artifact_path(
        prototype1_bundle,
        prototype1_config["data"]["stories_path"],
        "dataset/stories.jsonl",
    )
    stories = read_jsonl(stories_path)
    audit = validate_story_rows(stories, emotions, forbidden_terms)
    if audit["lexical_leakage"]:
        raise ValueError(f"Found forbidden emotion terms in Prototype 1 stories: {audit['lexical_leakage'][:5]}")

    train_topics = set(prototype1_config["data"].get("train_topics", []))
    test_topics = set(prototype1_config["data"].get("test_topics", []))
    if not train_topics or not test_topics:
        train_topics, test_topics = split_topics(
            [row["topic"] for row in stories],
            float(prototype1_config["data"]["train_topic_fraction"]),
            int(prototype1_config["seed"]),
        )
    train_rows = [row for row in stories if row["topic"] in train_topics]
    test_rows = [row for row in stories if row["topic"] in test_topics]

    model_config = {**prototype1_config["model"], **config.get("model", {})}
    device = choose_device(model_config["device"])
    dtype = choose_dtype(model_config["dtype"], device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        revision=model_config.get("revision", "main"),
        local_files_only=bool(model_config.get("local_files_only", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        revision=model_config.get("revision", "main"),
        dtype=dtype,
        local_files_only=bool(model_config.get("local_files_only", False)),
    ).to(device)
    model.eval()

    number_of_layers = len(decoder_layers(model))
    layer_indices = resolve_layer_indices(model_config.get("layers", "all"), number_of_layers)
    clean_vectors, vector_layers = load_emotion_vectors(
        prototype1_bundle / config["prototype1"].get("emotion_vectors", "emotion_vectors.safetensors"),
        emotions,
    )
    neutral_pca_path = prototype1_bundle / config["prototype1"].get(
        "neutral_pca", "neutral_pca.safetensors"
    )
    neutral_pca_metadata = safetensor_metadata(neutral_pca_path)
    if vector_layers != layer_indices:
        selected = [vector_layers.index(layer) for layer in layer_indices if layer in vector_layers]
        if len(selected) != len(layer_indices):
            raise ValueError("Configured layers are not all present in Prototype 1 vectors")
        clean_vectors = clean_vectors[:, selected]

    extraction = {**prototype1_config["extraction"], **config.get("extraction", {})}
    token_start = int(extraction.get("token_start", 50))
    activation_args = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "layer_indices": layer_indices,
        "token_start": token_start,
        "batch_size": int(extraction.get("batch_size", 2)),
    }
    train_activations = pooled_activations(
        texts=[row["text"] for row in train_rows], **activation_args
    )
    test_activations = pooled_activations(texts=[row["text"] for row in test_rows], **activation_args)

    raw_vectors = difference_in_means(
        train_activations, [row["emotion"] for row in train_rows], emotions
    )
    raw_vectors = raw_vectors / raw_vectors.norm(dim=-1, keepdim=True)

    labels = [row["emotion"] for row in test_rows]
    raw_validation = evaluate_vectors(test_activations, labels, emotions, raw_vectors)
    clean_validation = evaluate_vectors(test_activations, labels, emotions, clean_vectors)
    raw_scores = score_layers(test_activations, raw_vectors)
    clean_scores = score_layers(test_activations, clean_vectors)

    selected_layer = int(
        prototype1_metrics.get(
            "selected_layer",
            resolve_evaluation_layer(model_config.get("evaluation_layer", "two_thirds"), number_of_layers),
        )
    )
    if selected_layer not in layer_indices:
        raise ValueError("Prototype 1 selected layer must be included in Prototype 2 layers")

    lexical_rows = config.get("lexical_robustness", {}).get("scenarios") or default_lexical_scenarios()
    intensity_rows = config.get("intensity_sweep", {}).get("scenarios") or default_intensity_scenarios(emotions)
    lexical_activations = pooled_activations(
        texts=[row["text"] for row in lexical_rows], **activation_args
    )
    intensity_activations = pooled_activations(
        texts=[row["text"] for row in intensity_rows], **activation_args
    )

    pca_rows = pca_comparison(raw_validation, clean_validation, layer_indices)
    calibration_rows = calibration_diagnostic(
        train_activations,
        test_activations,
        [row["emotion"] for row in train_rows],
        labels,
        emotions,
        clean_vectors,
        layer_indices,
    )
    lexical = lexical_diagnostic(
        lexical_rows, lexical_activations, emotions, clean_vectors, layer_indices, forbidden_terms
    )
    intensity = intensity_diagnostic(
        intensity_rows,
        intensity_activations,
        emotions,
        clean_vectors,
        layer_indices,
        forbidden_terms,
    )
    for row in lexical["layers"]:
        row["chance_accuracy"] = lexical["chance_accuracy"]
    logit_lens_config = config.get("logit_lens", {})
    logit_lens = logit_lens_diagnostic(
        model,
        tokenizer,
        test_activations,
        labels,
        emotions,
        layer_indices,
        terms_by_emotion=logit_lens_config.get("terms"),
        batch_size=int(logit_lens_config.get("batch_size", extraction.get("batch_size", 2))),
    )

    metrics = {
        "all_soft_gates_pass": None,
        "selected_layer": selected_layer,
        "selection_metric": "Prototype 1 pre-registered layer; Prototype 2 layer sweep is diagnostic only",
        "prototype1_run_dir": str(prototype1_bundle),
        "prototype1_artifacts": {
            "emotion_vectors": str(
                prototype1_bundle
                / config["prototype1"].get("emotion_vectors", "emotion_vectors.safetensors")
            ),
            "neutral_pca": neutral_pca_metadata,
            "stories": str(stories_path),
        },
        "prototype1_summary": json.loads((prototype1_bundle / "summary.json").read_text()),
        "dataset_audit": audit,
        "split": {
            "train_topics": sorted(train_topics),
            "test_topics": sorted(test_topics),
            "train_stories": len(train_rows),
            "test_stories": len(test_rows),
        },
        "pooling": [
            pooling_diagnostics(tokenizer, train_rows, token_start, "prototype1_train"),
            pooling_diagnostics(tokenizer, test_rows, token_start, "prototype1_test"),
            pooling_diagnostics(tokenizer, lexical_rows, token_start, "lexical"),
            pooling_diagnostics(tokenizer, intensity_rows, token_start, "intensity"),
        ],
        "shuffled_label_control": [
            {**row, "layer": layer_indices[row["layer_position"]]}
            for row in shuffled_label_control(
                test_activations,
                labels,
                emotions,
                clean_vectors,
                int(config.get("controls", {}).get("shuffle_seed", seed + 1)),
            )
        ],
        "pca_comparison": pca_rows,
        "layer_sweep": {
            "raw": best_layers(pca_rows, "raw"),
            "pca_cleaned": best_layers(pca_rows, "pca_cleaned"),
            "diagnostic_only": True,
        },
        "emotion_confusion": confusion_diagnostics(
            clean_scores, clean_validation, emotions, layer_indices
        ),
        "cross_topic_generalization": {
            "raw": topic_stratified_metrics(test_rows, raw_scores, emotions, layer_indices),
            "pca_cleaned": topic_stratified_metrics(
                test_rows, clean_scores, emotions, layer_indices
            ),
            "diagnostic_only": True,
        },
        "logit_lens": logit_lens,
        "lexical_robustness": lexical,
        "intensity_sweep": intensity,
        "calibration": calibration_rows,
    }
    soft = {
        "real_macro_auc_exceeds_shuffled_at_selected_layer": selected_layer_row(
            metrics["shuffled_label_control"], selected_layer
        )["effect"]["macro_auc"]
        > 0,
        "mean_intensity_spearman_positive_at_selected_layer": selected_layer_row(
            metrics["intensity_sweep"]["layers"], selected_layer
        )["mean_spearman"]
        > 0,
        "lexical_accuracy_exceeds_chance_at_selected_layer": selected_layer_row(
            metrics["lexical_robustness"]["layers"], selected_layer
        )["metrics"]["accuracy"]
        > metrics["lexical_robustness"]["chance_accuracy"],
    }
    metrics["soft_gates"] = soft
    metrics["all_soft_gates_pass"] = all(soft.values())
    metrics["answers"] = answer_summary(metrics, selected_layer)

    created_at = datetime.now(timezone.utc)
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    resolved = json.loads(json.dumps(config))
    resolved["model"] = {
        **model_config,
        "resolved_device": device,
        "resolved_dtype": str(dtype),
        "resolved_layers": layer_indices,
        "number_of_layers": number_of_layers,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
    }
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
    dataset_output = output / "dataset"
    dataset_output.mkdir()
    shutil.copy2(stories_path, dataset_output / "stories.jsonl")
    write_jsonl(dataset_output / "lexical_scenarios.jsonl", lexical_rows)
    write_jsonl(dataset_output / "intensity_scenarios.jsonl", intensity_rows)
    diagnostics = output / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "lexical_scores.json").write_text(
        json.dumps(metrics["lexical_robustness"], indent=2) + "\n"
    )
    (diagnostics / "intensity_scores.json").write_text(
        json.dumps(metrics["intensity_sweep"], indent=2) + "\n"
    )
    (diagnostics / "logit_lens.json").write_text(
        json.dumps(metrics["logit_lens"], indent=2) + "\n"
    )
    (diagnostics / "cross_topic_generalization.json").write_text(
        json.dumps(metrics["cross_topic_generalization"], indent=2) + "\n"
    )

    code = git_metadata()
    environment = {
        "created_at": created_at.isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "model_name": model_config["name"],
        "requested_revision": model_config.get("revision", "main"),
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "code_git_commit": code["commit"],
        "code_git_dirty": code["dirty"],
        "config_sha256": config_hash,
        "stories_sha256": dataset_sha256(stories),
        "run_id": run_id,
    }
    manifest = build_manifest(
        run_id=run_id,
        created_at=created_at.isoformat(),
        config=resolved,
        resolved_model_revision=environment["resolved_model_revision"],
        code=code,
    )
    summary = {
        "all_soft_gates_pass": metrics["all_soft_gates_pass"],
        "selected_layer": selected_layer,
        "shuffled_macro_auc_effect": selected_layer_row(
            metrics["shuffled_label_control"], selected_layer
        )["effect"]["macro_auc"],
        "lexical_accuracy": selected_layer_row(
            metrics["lexical_robustness"]["layers"], selected_layer
        )["metrics"]["accuracy"],
        "intensity_mean_spearman": selected_layer_row(
            metrics["intensity_sweep"]["layers"], selected_layer
        )["mean_spearman"],
        "calibrated_margin_delta": selected_layer_row(
            metrics["calibration"], selected_layer
        )["calibrated_minus_uncalibrated"]["mean_correct_margin"],
        "answers": metrics["answers"],
        "logit_lens_selected_layer_accuracy": selected_layer_row(
            metrics["logit_lens"]["layers"], selected_layer
        )["metrics"]["accuracy"]
        if metrics["logit_lens"]["available"]
        else None,
        "minimum_topic_accuracy": selected_layer_row(
            metrics["cross_topic_generalization"]["pca_cleaned"]["layers"], selected_layer
        )["minimum_topic_accuracy"],
    }
    for filename, value in (
        ("config.json", resolved),
        ("metrics.json", metrics),
        ("environment.json", environment),
        ("manifest.json", manifest),
        ("summary.json", summary),
    ):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"output": str(output), **summary}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prototype 2 semantic diagnostics")
    parser.add_argument("--config", type=Path, default=Path("configs/prototype2.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import transformers
import yaml
from safetensors.torch import save_file
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .instrumentation import decoder_layers, resolve_layer_index
from .prototype0 import choose_device, choose_dtype
from .tracking import build_manifest, git_metadata, make_run_id, sha256_json


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    return hashlib.sha256(canonical.encode()).hexdigest()


def _forbidden_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    variants = [escaped]
    if term.endswith("y") and len(term) > 1:
        variants.append(re.escape(term[:-1]) + r"i(?:ly|ness|er|est)?")
    if term.endswith("e") and len(term) > 1:
        variants.append(re.escape(term[:-1]) + r"(?:ing|ed)")
    variants.append(escaped + r"(?:s|ed|ing|ly)?")
    return re.compile(rf"\b(?:{'|'.join(dict.fromkeys(variants))})\b", re.I)


def split_topics(topics: list[str], train_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    if len(topics) < 2:
        raise ValueError("At least two topics are required for a held-out topic split")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("data.train_topic_fraction must be between zero and one")
    ordered = sorted(set(topics))
    random.Random(seed).shuffle(ordered)
    cutoff = min(max(round(len(ordered) * train_fraction), 1), len(ordered) - 1)
    return set(ordered[:cutoff]), set(ordered[cutoff:])


def validate_story_rows(
    rows: list[dict[str, Any]],
    emotions: list[str],
    forbidden_terms: dict[str, list[str]],
) -> dict[str, Any]:
    required = {"topic", "emotion", "text"}
    invalid = [index for index, row in enumerate(rows) if not required <= row.keys()]
    if invalid:
        raise ValueError(f"Story rows missing required fields at indices: {invalid[:5]}")
    unknown = sorted({row["emotion"] for row in rows} - set(emotions))
    if unknown:
        raise ValueError(f"Unknown emotion labels in stories: {unknown}")

    leaks = []
    for index, row in enumerate(rows):
        terms = [row["emotion"], *forbidden_terms.get(row["emotion"], [])]
        matched = [term for term in terms if _forbidden_pattern(term).search(row["text"])]
        if matched:
            leaks.append({"row": index, "emotion": row["emotion"], "terms": matched})
    counts = Counter((row["topic"], row["emotion"]) for row in rows)
    return {
        "number_of_stories": len(rows),
        "number_of_topics": len({row["topic"] for row in rows}),
        "emotion_counts": dict(Counter(row["emotion"] for row in rows)),
        "minimum_stories_per_topic_emotion": min(counts.values(), default=0),
        "lexical_leakage": leaks,
    }


def _clean_generation(text: str) -> str:
    text = re.sub(r"^.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"^\s*(?:\[?story(?:\s+\d+)?\]?\s*[:.-]?)", "", text, flags=re.I)
    return text.strip().strip('"')


def _generation_malformed_reason(text: str) -> str | None:
    if re.search(r"<think\b", text, re.I) and not re.search(r"</think>", text, re.I):
        return "unterminated_think"
    return None


def _emotion_generation_guidance(emotion: str) -> str:
    guidance = {
        "happy": (
            "Use signs such as an easing posture, quickened steps, careful but buoyant "
            "choices, spontaneous generosity, or a private smile."
        ),
        "sad": (
            "Use signs such as slowed movement, quiet withdrawal, lingering over small "
            "objects, heavy pauses, or difficulty continuing ordinary tasks."
        ),
        "angry": (
            "Use signs such as clipped movements, tightened hands, controlled speech, "
            "sharp decisions, or forceful handling of ordinary objects."
        ),
        "afraid": (
            "Use signs such as scanning exits, guarded breathing, hesitation, checking "
            "details repeatedly, or keeping close to familiar things."
        ),
    }
    return guidance.get(
        emotion,
        "Use only concrete actions, choices, physical sensations, and body language.",
    )


def _generation_prompt(topic: str, emotion: str, forbidden: list[str], neutral: bool) -> str:
    if neutral:
        return (
            "Write one self-contained paragraph based on the premise below. Describe ordinary "
            "events factually, without emotional language, emotional reactions, or dramatic stakes.\n\n"
            f"Premise: {topic}\n\nReturn only the paragraph."
        )
    avoid = ", ".join([emotion, *forbidden])
    return (
        "Write one self-contained story paragraph based on the premise below. The paragraph "
        "should imply the private target state through concrete behavior, not labels. If the "
        "premise is ambiguous, add plausible context that makes the target state fit naturally. "
        f"{_emotion_generation_guidance(emotion)} "
        f"Forbidden words that must not appear in the paragraph: {avoid}. "
        "Do not name, summarize, or explain the state.\n\n"
        f"Premise: {topic}\n\nReturn only the story paragraph."
    )


def generation_inputs(
    tokenizer: Any,
    prompt: str,
    device: str,
    enable_thinking: bool = True,
) -> tuple[Tensor, Tensor]:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=enable_thinking,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        if isinstance(encoded, Tensor):
            input_ids = encoded.to(device)
            attention_mask = torch.ones_like(input_ids)
            return input_ids, attention_mask
        batch = encoded.to(device) if hasattr(encoded, "to") else encoded
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids.to(device), attention_mask.to(device)
    batch = tokenizer(prompt, return_tensors="pt")
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)


def generation_failure_report(
    *,
    topic: str,
    emotion: str,
    story_index: int,
    sequence: int,
    maximum_attempts: int,
    forbidden_terms: list[str],
    prompt: str,
    attempts: list[dict[str, Any]],
    successful_rows: int,
) -> dict[str, Any]:
    matched_counts = Counter(
        term for attempt in attempts for term in attempt.get("matched_terms", [])
    )
    return {
        "topic": topic,
        "emotion": emotion,
        "story_index": story_index,
        "sequence": sequence,
        "maximum_attempts": maximum_attempts,
        "successful_rows_before_failure": successful_rows,
        "forbidden_terms": [emotion, *forbidden_terms],
        "matched_term_counts": dict(matched_counts),
        "prompt": prompt,
        "attempts": attempts,
    }


def write_generation_failure_diagnostics(
    dataset_path: Path,
    report: dict[str, Any],
    partial_rows: list[dict[str, Any]],
) -> Path:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path = dataset_path.parent / "generation_failure.json"
    failure_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_jsonl(dataset_path.parent / "generation_partial_stories.jsonl", partial_rows)
    return failure_path


def generate_dataset(config: dict[str, Any]) -> Path:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generation = config["generation"]
    device = choose_device(generation["device"])
    dtype = choose_dtype(generation["dtype"], device)
    tokenizer = AutoTokenizer.from_pretrained(
        generation["model"],
        revision=generation.get("revision", "main"),
        local_files_only=bool(generation.get("local_files_only", False)),
    )
    model = AutoModelForCausalLM.from_pretrained(
        generation["model"],
        revision=generation.get("revision", "main"),
        dtype=dtype,
        local_files_only=bool(generation.get("local_files_only", False)),
    ).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = config["data"]
    emotions = list(data["emotions"])
    forbidden = data.get("forbidden_terms", {})
    stories_per_pair = int(generation["stories_per_topic_emotion"])
    maximum_attempts = int(generation.get("maximum_attempts_per_story", 3))
    dataset_path = Path(data["stories_path"])
    neutral_path = Path(data["neutral_path"])
    rows: list[dict[str, Any]] = []

    enable_thinking = bool(generation.get("enable_thinking", True))

    def sample(prompt: str, sample_seed: int) -> tuple[str, str | None, str]:
        torch.manual_seed(sample_seed)
        encoded, attention_mask = generation_inputs(tokenizer, prompt, device, enable_thinking)
        with torch.inference_mode():
            output = model.generate(
                input_ids=encoded,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=float(generation.get("temperature", 0.8)),
                top_p=float(generation.get("top_p", 0.95)),
                max_new_tokens=int(generation.get("max_new_tokens", 220)),
                pad_token_id=tokenizer.pad_token_id,
            )
        raw = tokenizer.decode(output[0, encoded.shape[1] :], skip_special_tokens=True)
        malformed_reason = _generation_malformed_reason(raw)
        if malformed_reason:
            return "", malformed_reason, raw
        return _clean_generation(raw), None, raw

    sequence = 0
    for topic in data["topics"]:
        for emotion in emotions:
            for story_index in range(stories_per_pair):
                prompt = _generation_prompt(topic, emotion, forbidden.get(emotion, []), False)
                text = ""
                attempts = []
                for attempt in range(maximum_attempts):
                    sample_seed = seed + sequence * maximum_attempts + attempt
                    text, malformed_reason, raw_text = sample(prompt, sample_seed)
                    check = validate_story_rows(
                        [{"topic": topic, "emotion": emotion, "text": text}],
                        emotions,
                        forbidden,
                    )
                    leakage = check["lexical_leakage"]
                    matched_terms = leakage[0]["terms"] if leakage else []
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "sample_seed": sample_seed,
                            "accepted": bool(text and not leakage and not malformed_reason),
                            "malformed_reason": malformed_reason,
                            "matched_terms": matched_terms,
                            "character_count": len(text),
                            "word_count": len(text.split()),
                            "text": text,
                            "raw_text": raw_text,
                        }
                    )
                    if text and not check["lexical_leakage"] and not malformed_reason:
                        break
                else:
                    report = generation_failure_report(
                        topic=topic,
                        emotion=emotion,
                        story_index=story_index,
                        sequence=sequence,
                        maximum_attempts=maximum_attempts,
                        forbidden_terms=forbidden.get(emotion, []),
                        prompt=prompt,
                        attempts=attempts,
                        successful_rows=len(rows),
                    )
                    failure_path = write_generation_failure_diagnostics(
                        dataset_path, report, rows
                    )
                    print(
                        json.dumps(
                            {
                                "generation_failure": str(failure_path),
                                "partial_stories": str(
                                    failure_path.parent / "generation_partial_stories.jsonl"
                                ),
                                "topic": topic,
                                "emotion": emotion,
                                "matched_term_counts": report["matched_term_counts"],
                            },
                            indent=2,
                        )
                    )
                    raise RuntimeError(
                        f"Could not generate a leakage-free story for {emotion!r}/{topic!r}; "
                        f"diagnostics written to {failure_path}"
                    )
                rows.append(
                    {
                        "id": f"story-{sequence:06d}",
                        "topic": topic,
                        "emotion": emotion,
                        "story_index": story_index,
                        "text": text,
                    }
                )
                sequence += 1

    neutral_rows = []
    for index, topic in enumerate(data["neutral_topics"]):
        prompt = _generation_prompt(topic, "", [], True)
        text, malformed_reason, _raw_text = sample(prompt, seed + sequence)
        if malformed_reason:
            raise RuntimeError(
                f"Could not generate a valid neutral paragraph for {topic!r}: {malformed_reason}"
            )
        neutral_rows.append(
            {"id": f"neutral-{index:06d}", "topic": topic, "text": text}
        )
        sequence += 1

    write_jsonl(dataset_path, rows)
    write_jsonl(neutral_path, neutral_rows)
    print(json.dumps({"stories": str(dataset_path), "neutral": str(neutral_path)}, indent=2))
    return dataset_path


def encode(tokenizer: Any, texts: list[str], device: str) -> dict[str, Tensor]:
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return {key: value.to(device) for key, value in batch.items()}


def flat_nonzero(values: Tensor) -> Tensor:
    return torch.nonzero(values, as_tuple=False).flatten()


def pooled_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: str,
    layer_indices: list[int],
    token_start: int,
    batch_size: int,
) -> Tensor:
    batches = []
    for offset in range(0, len(texts), batch_size):
        batch = encode(tokenizer, texts[offset : offset + batch_size], device)
        with torch.inference_mode():
            output = model(**batch, output_hidden_states=True, use_cache=False)
        masks = batch["attention_mask"].bool()
        pooled_layers = []
        for layer_index in layer_indices:
            hidden = output.hidden_states[layer_index + 1]
            vectors = []
            for row in range(hidden.shape[0]):
                valid = flat_nonzero(masks[row])
                selected = valid[token_start:] if len(valid) > token_start else valid[-1:]
                vectors.append(hidden[row, selected].float().mean(dim=0).cpu())
            pooled_layers.append(torch.stack(vectors))
        batches.append(torch.stack(pooled_layers, dim=1))
    return torch.cat(batches) if batches else torch.empty(0)


def pooling_diagnostics(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    token_start: int,
    split_name: str,
) -> dict[str, Any]:
    lengths = []
    fallback_rows = []
    per_emotion: dict[str, dict[str, int]] = {}
    for index, row in enumerate(rows):
        token_count = len(tokenizer(row["text"], truncation=True)["input_ids"])
        lengths.append(token_count)
        emotion = row.get("emotion")
        if emotion is not None:
            bucket = per_emotion.setdefault(emotion, {"rows": 0, "fallback_rows": 0})
            bucket["rows"] += 1
        if token_count <= token_start:
            fallback = {
                "row": index,
                "id": row.get("id"),
                "topic": row.get("topic"),
                "emotion": emotion,
                "token_count": token_count,
            }
            fallback_rows.append(fallback)
            if emotion is not None:
                per_emotion[emotion]["fallback_rows"] += 1
    return {
        "split": split_name,
        "rows": len(rows),
        "token_start": token_start,
        "minimum_tokens": min(lengths, default=0),
        "median_tokens": float(np.median(lengths)) if lengths else 0.0,
        "maximum_tokens": max(lengths, default=0),
        "final_token_fallback_rows": len(fallback_rows),
        "final_token_fallback_fraction": len(fallback_rows) / len(rows) if rows else 0.0,
        "fallback_examples": fallback_rows[:20],
        "per_emotion": per_emotion,
    }


def difference_in_means(activations: Tensor, labels: list[str], emotions: list[str]) -> Tensor:
    global_mean = activations.mean(dim=0)
    vectors = []
    for emotion in emotions:
        indices = [index for index, label in enumerate(labels) if label == emotion]
        if not indices:
            raise ValueError(f"No training stories for emotion {emotion!r}")
        vectors.append(activations[indices].mean(dim=0) - global_mean)
    return torch.stack(vectors)


def neutral_pca(activations: Tensor, variance_threshold: float) -> tuple[list[Tensor], list[dict[str, Any]]]:
    components = []
    metadata = []
    for layer_index in range(activations.shape[1]):
        centered = activations[:, layer_index] - activations[:, layer_index].mean(dim=0)
        _u, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        variance = singular.square()
        ratios = variance / variance.sum().clamp_min(torch.finfo(variance.dtype).eps)
        cumulative = ratios.cumsum(0)
        count = int(torch.searchsorted(cumulative, variance_threshold).item()) + 1
        count = min(count, vh.shape[0])
        components.append(vh[:count])
        metadata.append(
            {
                "number_of_components": count,
                "explained_variance": float(ratios[:count].sum()),
            }
        )
    return components, metadata


def remove_components(vectors: Tensor, components: list[Tensor]) -> Tensor:
    cleaned = vectors.clone()
    for layer_index, basis in enumerate(components):
        current = cleaned[:, layer_index]
        cleaned[:, layer_index] = current - (current @ basis.T) @ basis
    norms = cleaned.norm(dim=-1, keepdim=True)
    if torch.any(~torch.isfinite(norms)) or torch.any(norms == 0):
        raise ValueError("PCA removal produced a zero or non-finite emotion vector")
    return cleaned / norms


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


def binary_auc(scores: Tensor, positives: Tensor) -> float:
    positive_count = int(positives.sum())
    negative_count = len(positives) - positive_count
    if not positive_count or not negative_count:
        return math.nan
    ranks = _average_ranks(scores.double())
    positive_rank_sum = float(ranks[positives].sum())
    return (positive_rank_sum - positive_count * (positive_count - 1) / 2) / (
        positive_count * negative_count
    )


def evaluate_vectors(
    activations: Tensor, labels: list[str], emotions: list[str], vectors: Tensor
) -> list[dict[str, Any]]:
    results = []
    label_ids = torch.tensor([emotions.index(label) for label in labels])
    for layer_position in range(vectors.shape[1]):
        scores = activations[:, layer_position] @ vectors[:, layer_position].T
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
            binary_auc(scores[:, index], label_ids == index) for index in range(len(emotions))
        ]
        per_emotion = {
            emotion: {
                "accuracy": float((predicted[label_ids == index] == index).float().mean()),
                "auc": aucs[index],
            }
            for index, emotion in enumerate(emotions)
        }
        results.append(
            {
                "accuracy": float((predicted == label_ids).float().mean()),
                "macro_auc": float(np.nanmean(aucs)),
                "mean_correct_margin": float(margins.mean()),
                "confusion_matrix": {
                    "labels": emotions,
                    "rows_are_true_labels": confusion,
                },
                "per_emotion": per_emotion,
            }
        )
    return results


def selected_layer_examples(
    rows: list[dict[str, Any]],
    activations: Tensor,
    emotions: list[str],
    raw_vectors: Tensor,
    clean_vectors: Tensor,
    layer_position: int,
) -> list[dict[str, Any]]:
    raw_scores = activations[:, layer_position] @ raw_vectors[:, layer_position].T
    clean_scores = activations[:, layer_position] @ clean_vectors[:, layer_position].T
    examples = []
    for index, row in enumerate(rows):
        label_index = emotions.index(row["emotion"])
        raw_predicted = int(raw_scores[index].argmax())
        clean_predicted = int(clean_scores[index].argmax())
        clean_masked = clean_scores[index].clone()
        clean_masked[label_index] = -torch.inf
        examples.append(
            {
                "id": row.get("id"),
                "topic": row["topic"],
                "emotion": row["emotion"],
                "raw_prediction": emotions[raw_predicted],
                "pca_cleaned_prediction": emotions[clean_predicted],
                "pca_cleaned_correct": clean_predicted == label_index,
                "pca_cleaned_margin": float(clean_scores[index, label_index] - clean_masked.max()),
                "raw_scores": {
                    emotion: float(raw_scores[index, emotion_index])
                    for emotion_index, emotion in enumerate(emotions)
                },
                "pca_cleaned_scores": {
                    emotion: float(clean_scores[index, emotion_index])
                    for emotion_index, emotion in enumerate(emotions)
                },
            }
        )
    return examples


def resolve_layer_indices(specification: Any, number_of_layers: int) -> list[int]:
    if specification == "all":
        return list(range(number_of_layers))
    if not isinstance(specification, list) or not specification:
        return [resolve_layer_index(specification, number_of_layers)]
    return sorted({resolve_layer_index(value, number_of_layers) for value in specification})


def resolve_evaluation_layer(specification: int | str, number_of_layers: int) -> int:
    if specification == "two_thirds":
        return min(round(number_of_layers * 2 / 3), number_of_layers - 1)
    return resolve_layer_index(specification, number_of_layers)


def run(config: dict[str, Any]) -> Path:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = config["data"]
    stories = read_jsonl(Path(data["stories_path"]))
    neutral = read_jsonl(Path(data["neutral_path"]))
    emotions = list(data["emotions"])
    audit = validate_story_rows(stories, emotions, data.get("forbidden_terms", {}))
    if audit["lexical_leakage"]:
        raise ValueError(f"Found forbidden emotion terms in stories: {audit['lexical_leakage'][:5]}")
    train_topics, test_topics = split_topics(
        [row["topic"] for row in stories], float(data["train_topic_fraction"]), seed
    )
    train_rows = [row for row in stories if row["topic"] in train_topics]
    test_rows = [row for row in stories if row["topic"] in test_topics]
    for split_name, split_rows in (("train", train_rows), ("test", test_rows)):
        missing = set(emotions) - {row["emotion"] for row in split_rows}
        if missing:
            raise ValueError(f"{split_name} split is missing emotions: {sorted(missing)}")

    model_config = config["model"]
    device = choose_device(model_config["device"])
    dtype = choose_dtype(model_config["dtype"], device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        revision=model_config["revision"],
        local_files_only=bool(model_config.get("local_files_only", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        revision=model_config["revision"],
        dtype=dtype,
        local_files_only=bool(model_config.get("local_files_only", False)),
    ).to(device)
    model.eval()
    number_of_layers = len(decoder_layers(model))
    layer_indices = resolve_layer_indices(model_config.get("layers", "all"), number_of_layers)
    extraction = config["extraction"]
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
    neutral_activations = pooled_activations(
        texts=[row["text"] for row in neutral], **activation_args
    )
    raw_vectors = difference_in_means(
        train_activations, [row["emotion"] for row in train_rows], emotions
    )
    raw_vectors = raw_vectors / raw_vectors.norm(dim=-1, keepdim=True)
    components, pca_metadata = neutral_pca(
        neutral_activations, float(extraction.get("neutral_variance_threshold", 0.5))
    )
    clean_vectors = remove_components(raw_vectors, components)
    labels = [row["emotion"] for row in test_rows]
    raw_validation = evaluate_vectors(test_activations, labels, emotions, raw_vectors)
    clean_validation = evaluate_vectors(test_activations, labels, emotions, clean_vectors)

    selected_layer = resolve_evaluation_layer(
        model_config.get("evaluation_layer", "two_thirds"), number_of_layers
    )
    if selected_layer not in layer_indices:
        raise ValueError("model.evaluation_layer must be included in model.layers")
    selected_position = layer_indices.index(selected_layer)
    selected = clean_validation[selected_position]
    thresholds = config["gates"]
    gates = {
        "topic_splits_are_disjoint": train_topics.isdisjoint(test_topics),
        "stories_have_no_label_leakage": not audit["lexical_leakage"],
        "held_out_accuracy_above_threshold": selected["accuracy"]
        >= float(thresholds["minimum_held_out_accuracy"]),
        "held_out_macro_auc_above_threshold": selected["macro_auc"]
        >= float(thresholds["minimum_held_out_macro_auc"]),
        "held_out_margin_is_positive": selected["mean_correct_margin"]
        >= float(thresholds.get("minimum_mean_correct_margin", 0.0)),
    }

    created_at = datetime.now(timezone.utc)
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    resolved = json.loads(json.dumps(config))
    resolved["model"].update(
        {
            "resolved_device": device,
            "resolved_dtype": str(dtype),
            "resolved_layers": layer_indices,
            "number_of_layers": number_of_layers,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
        }
    )
    resolved["data"]["train_topics"] = sorted(train_topics)
    resolved["data"]["test_topics"] = sorted(test_topics)
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
    shutil.copy2(Path(data["stories_path"]), dataset_output / "stories.jsonl")
    shutil.copy2(Path(data["neutral_path"]), dataset_output / "neutral.jsonl")
    metrics = {
        "all_hard_gates_pass": all(gates.values()),
        "gates": gates,
        "selected_layer": selected_layer,
        "selection_metric": "pre_registered_model.evaluation_layer",
        "chance_accuracy": 1 / len(emotions),
        "dataset_audit": audit,
        "dataset_artifacts": {
            "stories": "dataset/stories.jsonl",
            "neutral": "dataset/neutral.jsonl",
        },
        "split": {
            "train_topics": sorted(train_topics),
            "test_topics": sorted(test_topics),
            "train_stories": len(train_rows),
            "test_stories": len(test_rows),
        },
        "pooling": [
            pooling_diagnostics(tokenizer, train_rows, token_start, "train"),
            pooling_diagnostics(tokenizer, test_rows, token_start, "test"),
            pooling_diagnostics(tokenizer, neutral, token_start, "neutral"),
        ],
        "neutral_pca": [
            {"layer": layer, **metadata}
            for layer, metadata in zip(layer_indices, pca_metadata, strict=True)
        ],
        "selected_layer_examples": selected_layer_examples(
            test_rows,
            test_activations,
            emotions,
            raw_vectors,
            clean_vectors,
            selected_position,
        ),
        "layers": [
            {
                "layer": layer,
                "raw": raw,
                "pca_cleaned": clean,
            }
            for layer, raw, clean in zip(
                layer_indices, raw_validation, clean_validation, strict=True
            )
        ],
    }
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
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "code_git_commit": code["commit"],
        "code_git_dirty": code["dirty"],
        "config_sha256": config_hash,
        "stories_sha256": dataset_sha256(stories),
        "neutral_sha256": dataset_sha256(neutral),
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
        "all_hard_gates_pass": metrics["all_hard_gates_pass"],
        "selected_layer": selected_layer,
        "held_out_accuracy": selected["accuracy"],
        "held_out_macro_auc": selected["macro_auc"],
        "mean_correct_margin": selected["mean_correct_margin"],
        "number_of_emotions": len(emotions),
    }
    for filename, value in (
        ("config.json", resolved),
        ("metrics.json", metrics),
        ("environment.json", environment),
        ("manifest.json", manifest),
        ("summary.json", summary),
    ):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n")
    vector_tensors = {
        f"{emotion}/layer_{layer}": clean_vectors[emotion_index, layer_position].contiguous()
        for emotion_index, emotion in enumerate(emotions)
        for layer_position, layer in enumerate(layer_indices)
    }
    pca_tensors = {
        f"layer_{layer}": basis.contiguous()
        for layer, basis in zip(layer_indices, components, strict=True)
    }
    save_file(vector_tensors, output / "emotion_vectors.safetensors")
    save_file(pca_tensors, output / "neutral_pca.safetensors")
    print(json.dumps({"output": str(output), **summary}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data or extract Prototype 1 emotion vectors")
    parser.add_argument("--config", type=Path, default=Path("configs/prototype1.yaml"))
    parser.add_argument("--stage", choices=("generate", "extract", "all"), default="extract")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in {"generate", "all"}:
        generate_dataset(config)
    if args.stage in {"extract", "all"}:
        run(config)


if __name__ == "__main__":
    main()

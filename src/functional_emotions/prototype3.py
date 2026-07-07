from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file
from torch import Tensor

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
        by_layer = []
        for layer in layers:
            key = f"{emotion}/layer_{layer}"
            if key not in tensors:
                raise ValueError(f"Missing vector tensor {key!r}")
            by_layer.append(tensors[key].float())
        vectors.append(torch.stack(by_layer))
    return torch.stack(vectors), layers


def cosine_matrix(vectors: Tensor) -> Tensor:
    normalized = torch.nn.functional.normalize(vectors.float(), dim=-1)
    return normalized @ normalized.T


def upper_triangle_values(matrix: Tensor) -> Tensor:
    indices = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1)
    return matrix[indices[0], indices[1]]


def average_ranks(values: Tensor) -> Tensor:
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


def pearson(x: Tensor, y: Tensor) -> float:
    x = x.double()
    y = y.double()
    if len(x) < 2:
        return math.nan
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.linalg.norm(x_centered) * torch.linalg.norm(y_centered)
    if float(denominator) == 0.0:
        return math.nan
    return float((x_centered @ y_centered) / denominator)


def spearman(x: Tensor, y: Tensor) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def pca_projection(vectors: Tensor, dimensions: int = 2) -> dict[str, Any]:
    centered = vectors.float() - vectors.float().mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    usable = min(dimensions, vh.shape[0])
    coordinates = centered @ vh[:usable].T
    total_variance = float((singular_values**2).sum())
    explained = []
    for value in singular_values[:usable]:
        explained.append(float((value**2) / total_variance) if total_variance else 0.0)
    while coordinates.shape[1] < dimensions:
        coordinates = torch.cat([coordinates, torch.zeros(coordinates.shape[0], 1)], dim=1)
        explained.append(0.0)
    return {
        "coordinates": coordinates[:, :dimensions].tolist(),
        "explained_variance_ratio": explained[:dimensions],
    }


def nearest_neighbors(cosine: Tensor, emotions: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for index, emotion in enumerate(emotions):
        row = cosine[index].clone()
        row[index] = -torch.inf
        neighbor_index = int(torch.argmax(row))
        result[emotion] = {
            "emotion": emotions[neighbor_index],
            "cosine": float(row[neighbor_index]),
        }
    return result


def agglomerative_clusters(cosine: Tensor, emotions: list[str]) -> list[dict[str, Any]]:
    clusters = [[emotion] for emotion in emotions]
    emotion_index = {emotion: index for index, emotion in enumerate(emotions)}
    merges = []
    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_similarity = -math.inf
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                similarities = [
                    float(cosine[emotion_index[a], emotion_index[b]])
                    for a in clusters[left]
                    for b in clusters[right]
                ]
                similarity = sum(similarities) / len(similarities)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (left, right)
        assert best_pair is not None
        left, right = best_pair
        merged = sorted(clusters[left] + clusters[right])
        merges.append(
            {
                "left": clusters[left],
                "right": clusters[right],
                "merged": merged,
                "average_cosine": best_similarity,
            }
        )
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left, right}
        ] + [merged]
    return merges


def pairwise_rating_distances(emotions: list[str], ratings: dict[str, dict[str, float]], axis: str) -> Tensor:
    values = []
    for left in range(len(emotions)):
        for right in range(left + 1, len(emotions)):
            values.append(abs(float(ratings[emotions[left]][axis]) - float(ratings[emotions[right]][axis])))
    return torch.tensor(values, dtype=torch.float64)


def valence_arousal_alignment(
    cosine: Tensor, emotions: list[str], ratings: dict[str, dict[str, float]]
) -> dict[str, Any]:
    similarities = upper_triangle_values(cosine).double()
    return {
        axis: {
            "spearman_similarity_vs_negative_distance": spearman(
                similarities,
                -pairwise_rating_distances(emotions, ratings, axis),
            ),
            "pearson_similarity_vs_negative_distance": pearson(
                similarities,
                -pairwise_rating_distances(emotions, ratings, axis),
            ),
        }
        for axis in ("valence", "arousal")
    }


def layer_geometry(
    vectors: Tensor,
    layers: list[int],
    emotions: list[str],
    ratings: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    for position, layer in enumerate(layers):
        layer_vectors = vectors[:, position]
        cosine = cosine_matrix(layer_vectors)
        pca = pca_projection(layer_vectors)
        rows.append(
            {
                "layer": layer,
                "cosine_matrix": {
                    "labels": emotions,
                    "values": cosine.tolist(),
                },
                "mean_abs_off_diagonal_cosine": float(
                    upper_triangle_values(cosine).abs().mean()
                ),
                "nearest_neighbors": nearest_neighbors(cosine, emotions),
                "agglomerative_clusters": agglomerative_clusters(cosine, emotions),
                "pca": {
                    "labels": emotions,
                    **pca,
                },
                "valence_arousal_alignment": valence_arousal_alignment(
                    cosine, emotions, ratings
                ),
            }
        )
    return rows


def representational_similarity(layers: list[dict[str, Any]], selected_layer: int) -> dict[str, Any]:
    flattened = {
        row["layer"]: upper_triangle_values(torch.tensor(row["cosine_matrix"]["values"]))
        for row in layers
    }
    selected = flattened[selected_layer]
    by_selected = [
        {
            "layer": layer,
            "pearson_to_selected_layer": pearson(values, selected),
            "spearman_to_selected_layer": spearman(values, selected),
        }
        for layer, values in flattened.items()
    ]
    adjacent = []
    ordered_layers = [row["layer"] for row in layers]
    for left, right in zip(ordered_layers, ordered_layers[1:], strict=False):
        adjacent.append(
            {
                "left_layer": left,
                "right_layer": right,
                "pearson": pearson(flattened[left], flattened[right]),
                "spearman": spearman(flattened[left], flattened[right]),
            }
        )
    return {
        "selected_layer": selected_layer,
        "layers": by_selected,
        "adjacent_layers": adjacent,
    }


def selected_layer_from_bundle(config: dict[str, Any], metrics: dict[str, Any]) -> int:
    configured = config.get("geometry", {}).get("selected_layer")
    if configured is not None:
        return int(configured)
    if metrics.get("selected_layer") is None:
        raise ValueError("geometry.selected_layer is required when Prototype 1 metrics omit it")
    return int(metrics["selected_layer"])


def summary_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    selected = next(
        row for row in metrics["layers"] if row["layer"] == metrics["selected_layer"]
    )
    alignment = selected["valence_arousal_alignment"]
    return {
        "selected_layer": metrics["selected_layer"],
        "number_of_emotions": len(metrics["emotions"]),
        "number_of_layers": len(metrics["layers"]),
        "selected_layer_mean_abs_off_diagonal_cosine": selected[
            "mean_abs_off_diagonal_cosine"
        ],
        "selected_layer_valence_spearman": alignment["valence"][
            "spearman_similarity_vs_negative_distance"
        ],
        "selected_layer_arousal_spearman": alignment["arousal"][
            "spearman_similarity_vs_negative_distance"
        ],
        "selected_layer_nearest_neighbors": selected["nearest_neighbors"],
        "umap_available": metrics["umap"]["available"],
        "prototype4_recommendation": metrics["prototype4_recommendation"],
    }


def run(config: dict[str, Any]) -> Path:
    prototype1_bundle = Path(config["prototype1"]["run_dir"])
    prototype1_config = json.loads((prototype1_bundle / "config.json").read_text())
    prototype1_metrics = json.loads((prototype1_bundle / "metrics.json").read_text())
    prototype1_summary = json.loads((prototype1_bundle / "summary.json").read_text())
    prototype1_environment = json.loads((prototype1_bundle / "environment.json").read_text())
    emotions = list(prototype1_config["data"]["emotions"])
    ratings = config["valence_arousal"]["ratings"]
    missing_ratings = [emotion for emotion in emotions if emotion not in ratings]
    if missing_ratings:
        raise ValueError(f"Missing valence/arousal ratings for: {', '.join(missing_ratings)}")

    vector_path = resolve_artifact_path(
        prototype1_bundle,
        config["prototype1"].get("emotion_vectors", "emotion_vectors.safetensors"),
        "emotion_vectors.safetensors",
    )
    vectors, layers = load_emotion_vectors(vector_path, emotions)
    selected_layer = selected_layer_from_bundle(config, prototype1_metrics)
    if selected_layer not in layers:
        raise ValueError("Selected layer is not present in the emotion vector bundle")

    geometry_rows = layer_geometry(vectors, layers, emotions, ratings)
    rsa = representational_similarity(geometry_rows, selected_layer)
    metrics = {
        "selected_layer": selected_layer,
        "selection_metric": "Prototype 1 pre-registered layer; Prototype 3 layer sweep is diagnostic only",
        "emotions": emotions,
        "prototype1_run_dir": str(prototype1_bundle),
        "prototype1_summary": prototype1_summary,
        "prototype2_validation_run_dir": config.get("prototype2_validation", {}).get("run_dir"),
        "artifacts": {
            "emotion_vectors": str(vector_path),
        },
        "layers": geometry_rows,
        "representational_similarity": rsa,
        "umap": {
            "available": False,
            "reason": "UMAP is not a project dependency; PCA geometry is recorded and UMAP can be added as an optional diagnostic later.",
        },
        "interpretation_limits": [
            "Diagnostic geometry over compact four-emotion synthetic vectors.",
            "Does not establish paper-scale emotion space coverage.",
            "Does not test causal steering; Prototype 4 owns causal intervention.",
            "Does not make claims about subjective experience.",
        ],
        "prototype4_recommendation": "Proceed to causal steering only as a diagnostic follow-up over the canonical Prototype 2.5 vectors.",
    }

    created_at = datetime.now(timezone.utc)
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    resolved = json.loads(json.dumps(config))
    resolved["model"] = {
        "name": prototype1_config["model"]["name"],
        "requested_revision": prototype1_config["model"].get("revision"),
        "resolved_revision": prototype1_environment.get("resolved_model_revision"),
        "resolved_layers": layers,
        "selected_layer": selected_layer,
    }
    resolved["prototype1"]["resolved_run_dir"] = str(prototype1_bundle)
    resolved["prototype1"]["resolved_emotion_vectors"] = str(vector_path)
    config_hash = sha256_json(resolved)
    run_id = make_run_id(
        timestamp=timestamp,
        experiment=resolved["experiment"],
        model_name=resolved["model"]["name"],
        seed=int(resolved["seed"]),
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
        "model_name": resolved["model"]["name"],
        "requested_revision": resolved["model"]["requested_revision"],
        "resolved_model_revision": resolved["model"]["resolved_revision"],
        "code_git_commit": code["commit"],
        "code_git_dirty": code["dirty"],
        "config_sha256": config_hash,
        "run_id": run_id,
    }
    manifest = build_manifest(
        run_id=run_id,
        created_at=created_at.isoformat(),
        config=resolved,
        resolved_model_revision=environment["resolved_model_revision"],
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
    selected = next(row for row in geometry_rows if row["layer"] == selected_layer)
    (diagnostics / "selected_layer_geometry.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    (diagnostics / "representational_similarity.json").write_text(
        json.dumps(rsa, indent=2) + "\n"
    )
    (diagnostics / "pca_by_layer.json").write_text(
        json.dumps(
            [
                {"layer": row["layer"], "pca": row["pca"]}
                for row in geometry_rows
            ],
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output": str(output), **summary}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prototype 3 emotion-space geometry")
    parser.add_argument("--config", type=Path, default=Path("configs/prototype3.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

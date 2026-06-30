import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from functional_emotions.prototype3 import (
    cosine_matrix,
    pca_projection,
    run,
    spearman,
    valence_arousal_alignment,
)


def test_cosine_matrix_preserves_signed_geometry():
    vectors = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    cosine = cosine_matrix(vectors)

    assert torch.allclose(torch.diag(cosine), torch.ones(3))
    assert cosine[0, 1] == 0.0
    assert cosine[0, 2] == -1.0


def test_pca_projection_returns_two_stable_dimensions():
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    projection = pca_projection(vectors)

    assert len(projection["coordinates"]) == 4
    assert len(projection["coordinates"][0]) == 2
    assert sum(projection["explained_variance_ratio"]) > 0.99


def test_valence_alignment_uses_pairwise_similarity_ordering():
    emotions = ["happy", "sad", "angry"]
    cosine = torch.tensor(
        [
            [1.0, -0.8, -0.7],
            [-0.8, 1.0, 0.9],
            [-0.7, 0.9, 1.0],
        ]
    )
    ratings = {
        "happy": {"valence": 1.0, "arousal": 0.5},
        "sad": {"valence": -1.0, "arousal": -0.2},
        "angry": {"valence": -0.9, "arousal": 0.8},
    }

    alignment = valence_arousal_alignment(cosine, emotions, ratings)

    assert alignment["valence"]["spearman_similarity_vs_negative_distance"] > 0.5
    assert spearman(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])) == pytest.approx(1.0)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def test_prototype3_run_writes_geometry_artifacts(tmp_path):
    bundle = tmp_path / "prototype1"
    bundle.mkdir()
    emotions = ["happy", "sad", "angry"]
    tensors = {
        "happy/layer_0": torch.tensor([1.0, 0.0, 0.0]),
        "sad/layer_0": torch.tensor([-1.0, 0.0, 0.0]),
        "angry/layer_0": torch.tensor([-0.8, 0.2, 0.0]),
        "happy/layer_1": torch.tensor([0.8, 0.2, 0.0]),
        "sad/layer_1": torch.tensor([-0.8, 0.1, 0.0]),
        "angry/layer_1": torch.tensor([-0.7, 0.3, 0.0]),
    }
    save_file(tensors, bundle / "emotion_vectors.safetensors")
    write_json(
        bundle / "config.json",
        {
            "experiment": "prototype1-test",
            "seed": 7,
            "model": {"name": "org/model", "revision": "main"},
            "data": {"emotions": emotions},
        },
    )
    write_json(bundle / "metrics.json", {"selected_layer": 1})
    write_json(bundle / "summary.json", {"all_hard_gates_pass": True})
    write_json(
        bundle / "environment.json",
        {
            "created_at": "2026-06-29T12:00:00+00:00",
            "resolved_model_revision": "abc123",
        },
    )
    config = {
        "experiment": "prototype3-test",
        "seed": 42,
        "prototype1": {
            "run_dir": str(bundle),
            "emotion_vectors": "emotion_vectors.safetensors",
        },
        "prototype2_validation": {"run_dir": None},
        "geometry": {"selected_layer": None},
        "valence_arousal": {
            "ratings": {
                "happy": {"valence": 1.0, "arousal": 0.6},
                "sad": {"valence": -0.9, "arousal": -0.4},
                "angry": {"valence": -0.8, "arousal": 0.8},
            }
        },
        "output_dir": str(tmp_path / "prototype3"),
    }

    output = run(config)

    metrics = json.loads((output / "metrics.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert metrics["selected_layer"] == 1
    assert metrics["emotions"] == emotions
    assert summary["number_of_layers"] == 2
    assert (output / "diagnostics" / "selected_layer_geometry.json").is_file()
    assert (output / "manifest.json").is_file()


def test_prototype3_config_points_to_canonical_handoff():
    config = yaml.safe_load(Path("configs/prototype3.yaml").read_text())

    assert config["experiment"] == "prototype3"
    assert "colab-run-prototype-2.5" in config["prototype1"]["run_dir"]
    assert config["prototype1"]["emotion_vectors"] == "emotion_vectors.safetensors"
    assert set(config["valence_arousal"]["ratings"]) == {"happy", "sad", "angry", "afraid"}

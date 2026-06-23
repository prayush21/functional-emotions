import json
from pathlib import Path

from functional_emotions.tracking import make_run_id, register_bundle, sha256_json, summary_from_metrics


def sample_bundle(path: Path) -> None:
    config = {
        "experiment": "prototype0",
        "seed": 42,
        "model": {"name": "org/model", "revision": "main"},
    }
    environment = {"created_at": "2026-06-20T12:00:00+00:00"}
    metrics = {
        "all_hard_gates_pass": True,
        "layer_index": 2,
        "hidden_size": 8,
        "effect_monotonic_spearman": 1.0,
        "sweep": [
            {"raw_strength": -0.1, "positive_minus_negative_logit_margin": -1.0},
            {"raw_strength": 0.0, "positive_minus_negative_logit_margin": 0.0},
            {"raw_strength": 0.1, "positive_minus_negative_logit_margin": 1.0},
        ],
    }
    for name, value in (
        ("config.json", config),
        ("environment.json", environment),
        ("metrics.json", metrics),
    ):
        (path / name).write_text(json.dumps(value))


def test_run_id_is_stable():
    value = make_run_id(
        timestamp="20260620T120000Z",
        experiment="Prototype 0",
        model_name="Org/Model",
        seed=42,
        config_hash=sha256_json({"a": 1}),
    )
    assert value.startswith("20260620T120000Z__prototype-0__org-model__seed-42__")


def test_register_legacy_bundle_is_idempotent(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    sample_bundle(bundle)
    registry = tmp_path / "results"

    first = register_bundle(bundle, registry)
    second = register_bundle(bundle, registry)

    assert first == second
    assert (first / "manifest.json").is_file()
    assert (first / "summary.json").is_file()
    assert len(registry.joinpath("index.jsonl").read_text().splitlines()) == 1


def test_summary_supports_prototype1_metrics():
    metrics = {
        "all_hard_gates_pass": True,
        "selected_layer": 4,
        "layers": [
            {
                "layer": 4,
                "pca_cleaned": {
                    "accuracy": 0.75,
                    "macro_auc": 0.9,
                    "mean_correct_margin": 1.2,
                    "per_emotion": {"happy": {}, "sad": {}},
                },
            }
        ],
    }

    assert summary_from_metrics(metrics) == {
        "all_hard_gates_pass": True,
        "selected_layer": 4,
        "held_out_accuracy": 0.75,
        "held_out_macro_auc": 0.9,
        "mean_correct_margin": 1.2,
        "number_of_emotions": 2,
    }

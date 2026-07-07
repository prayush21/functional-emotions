from pathlib import Path

import yaml

from functional_emotions.prototype25 import (
    complete_prototype1_runs,
    dataset_plan,
    deep_merge,
    resolved_prototype1_config,
    resolved_prototype2_config,
)


def test_deep_merge_preserves_nested_base_values():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 4}}

    merged = deep_merge(base, override)

    assert merged == {"a": {"b": 4, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1


def test_prototype25_dataset_plan_counts_balanced_story_grid():
    config = yaml.safe_load(Path("configs/prototype25.yaml").read_text())
    plan = dataset_plan(config)

    assert plan["emotions"] == ["happy", "sad", "angry", "afraid"]
    assert plan["topics"] == 16
    assert plan["stories_per_topic_emotion"] == 4
    assert plan["total_emotion_stories"] == 256


def test_resolved_prototype25_configs_write_inside_orchestrator_output(tmp_path):
    config = yaml.safe_load(Path("configs/prototype25.yaml").read_text())
    output = tmp_path / "run"

    p1 = resolved_prototype1_config(config, output)
    p2 = resolved_prototype2_config(config, output, output / "prototype1" / "finished")

    assert p1["experiment"] == "prototype1-prototype25"
    assert p1["data"]["stories_path"] == str(output / "dataset" / "stories.jsonl")
    assert p1["output_dir"] == str(output / "prototype1")
    assert p2["experiment"] == "prototype2-prototype25-validation"
    assert p2["prototype1"]["run_dir"] == str(output / "prototype1" / "finished")
    assert p2["output_dir"] == str(output / "prototype2")


def test_complete_prototype1_runs_detects_finished_nested_outputs(tmp_path):
    complete = tmp_path / "prototype1" / "complete-run"
    incomplete = tmp_path / "prototype1" / "incomplete-run"
    for path in (
        complete / "dataset",
        incomplete,
    ):
        path.mkdir(parents=True)
    for name in (
        "config.json",
        "metrics.json",
        "emotion_vectors.safetensors",
        "neutral_pca.safetensors",
        "dataset/stories.jsonl",
    ):
        (complete / name).write_text("{}")
    (incomplete / "config.json").write_text("{}")

    assert complete_prototype1_runs(tmp_path / "prototype1") == [complete]

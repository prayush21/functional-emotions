from pathlib import Path

import yaml


def test_colab_config_targets_qwen_1_7b_cuda_float16():
    config = yaml.safe_load(Path("configs/prototype0_qwen3_1.7b_colab.yaml").read_text())

    assert config["model"]["name"] == "Qwen/Qwen3-1.7B-Base"
    assert config["model"]["device"] == "cuda"
    assert config["model"]["dtype"] == "float16"
    assert 0.0 in config["steering"]["strengths"]
    assert config["output_dir"] == "artifacts/prototype0-qwen3-1.7b-colab"


def test_prototype1_config_has_held_out_topics_and_paper_pooling():
    config = yaml.safe_load(Path("configs/prototype1.yaml").read_text())

    assert config["model"]["evaluation_layer"] == "two_thirds"
    assert config["extraction"]["token_start"] == 50
    assert config["extraction"]["neutral_variance_threshold"] == 0.5
    assert 0.0 < config["data"]["train_topic_fraction"] < 1.0
    assert len(config["data"]["topics"]) >= 2
    assert len(config["data"]["emotions"]) >= 2


def test_prototype2_config_points_to_prototype1_bundle_and_keeps_controls_diagnostic():
    config = yaml.safe_load(Path("configs/prototype2.yaml").read_text())

    assert config["experiment"] == "prototype2"
    assert config["prototype1"]["run_dir"]
    assert config["extraction"]["token_start"] == 50
    assert config["controls"]["shuffle_seed"] != config["seed"]


def test_prototype25_config_expands_balanced_extraction_before_geometry():
    config = yaml.safe_load(Path("configs/prototype25.yaml").read_text())
    overrides = config["prototype1"]["overrides"]

    assert config["experiment"] == "prototype2.5"
    assert overrides["generation"]["stories_per_topic_emotion"] > 3
    assert len(overrides["data"]["topics"]) > 12
    assert "afraid" in overrides["data"]["emotions"]
    assert config["prototype2"]["base_config"] == "configs/prototype2.yaml"

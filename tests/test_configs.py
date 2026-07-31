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


SCALED_BUNDLE = (
    "results/runs/20260710T223434Z__prototype2-5-qwen3-1-7b__qwen-qwen3-1-7b-base"
    "__seed-42__530f4b779c"
)


def test_downstream_1_7b_configs_change_only_labels_bundles_and_p51_mode():
    for name in ("prototype3", "prototype4", "prototype51"):
        base = yaml.safe_load(Path(f"configs/{name}.yaml").read_text())
        scaled = yaml.safe_load(Path(f"configs/{name}_qwen3_1.7b.yaml").read_text())

        assert scaled["experiment"] == f"{name}-qwen3-1.7b"
        assert scaled["prototype1"]["run_dir"].startswith(f"{SCALED_BUNDLE}/prototype1/")
        assert "qwen3-1-7b-base" in scaled["prototype1"]["run_dir"]
        base["experiment"] = scaled["experiment"]
        base["prototype1"]["run_dir"] = scaled["prototype1"]["run_dir"]
        if name == "prototype3":
            assert scaled["prototype2_validation"]["run_dir"].startswith(
                f"{SCALED_BUNDLE}/prototype2/"
            )
            base["prototype2_validation"]["run_dir"] = scaled["prototype2_validation"][
                "run_dir"
            ]
        if name == "prototype51":
            # The only other pre-registered change: the length-normalized
            # diagnostic mode is added; the primary mode is unchanged.
            assert scaled["scoring"]["primary"] == "option_text_logprob_margin"
            assert scaled["scoring"]["modes"] == [
                "choice_token_margin",
                "option_text_logprob_margin",
                "option_text_mean_logprob_margin",
            ]
            base["scoring"]["modes"] = scaled["scoring"]["modes"]
        assert scaled == base


def test_downstream_1_7b_configs_point_to_registered_bundle_files():
    for name in ("prototype3", "prototype4", "prototype51"):
        scaled = yaml.safe_load(Path(f"configs/{name}_qwen3_1.7b.yaml").read_text())
        bundle = Path(scaled["prototype1"]["run_dir"])
        assert (bundle / "config.json").is_file()
        assert (bundle / "metrics.json").is_file()
        assert (bundle / "emotion_vectors.safetensors").is_file()


def test_prototype25_1_7b_config_changes_only_extraction_model():
    base = yaml.safe_load(Path("configs/prototype25.yaml").read_text())
    scaled = yaml.safe_load(Path("configs/prototype25_qwen3_1.7b.yaml").read_text())

    assert scaled["experiment"] == "prototype2.5-qwen3-1.7b"
    assert scaled["prototype1"]["overrides"]["model"]["name"] == "Qwen/Qwen3-1.7B-Base"
    # Everything except the experiment label and extraction model name must
    # match the accepted 0.6B pre-registration exactly.
    base["experiment"] = scaled["experiment"]
    base["prototype1"]["overrides"]["model"]["name"] = "Qwen/Qwen3-1.7B-Base"
    assert scaled == base

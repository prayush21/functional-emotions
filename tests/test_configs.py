from pathlib import Path

import yaml


def test_colab_config_targets_qwen_1_7b_cuda_float16():
    config = yaml.safe_load(Path("configs/prototype0_qwen3_1.7b_colab.yaml").read_text())

    assert config["model"]["name"] == "Qwen/Qwen3-1.7B-Base"
    assert config["model"]["device"] == "cuda"
    assert config["model"]["dtype"] == "float16"
    assert 0.0 in config["steering"]["strengths"]
    assert config["output_dir"] == "artifacts/prototype0-qwen3-1.7b-colab"

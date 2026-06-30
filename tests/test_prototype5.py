import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml
from safetensors.torch import save_file

from functional_emotions import prototype5
from functional_emotions.prototype5 import (
    INTERPRETATION_CAVEAT,
    aggregate_preference_rows,
    choice_token_ids,
    expected_direction,
    parse_activity_pairs,
    preference_margin,
    run,
    zero_fidelity,
)


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.vocab = {
            "<pad>": 0,
            "<eos>": 1,
            "neutral": 2,
            "The": 3,
            "A": 20,
            "B": 21,
        }

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        token = text.strip().split()[0] if text.strip() else "neutral"
        return [self.vocab.get(token, 2)]

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False, **_kwargs):
        _ = truncation
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(text) for text in texts]
        max_length = max(len(row) for row in encoded)
        if padding:
            encoded = [row + [self.pad_token_id] * (max_length - len(row)) for row in encoded]
        attention = [
            [0 if token == self.pad_token_id else 1 for token in row] for row in encoded
        ]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        return {"input_ids": encoded, "attention_mask": attention}


class FakeLayer(nn.Module):
    def forward(self, hidden):
        return hidden


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(), FakeLayer()])


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_commit_hash="fake-revision", hidden_size=4)
        self.model = FakeBackbone()
        self.embedding = nn.Embedding(32, 4)
        self.lm_head = nn.Linear(4, 32, bias=False)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.lm_head.weight.zero_()
            self.lm_head.weight[20, :] = 1.0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def forward(self, input_ids, attention_mask=None):
        _ = attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "prototype1"
    bundle.mkdir()
    emotions = ["happy", "sad", "angry", "afraid"]
    basis = torch.eye(4)
    tensors = {}
    for layer in (0, 1):
        for index, emotion in enumerate(emotions):
            tensors[f"{emotion}/layer_{layer}"] = basis[index].clone()
    save_file(tensors, bundle / "emotion_vectors.safetensors")
    write_json(
        bundle / "config.json",
        {
            "experiment": "prototype1-test",
            "seed": 7,
            "model": {
                "name": "fake/model",
                "revision": "main",
                "dtype": "float32",
                "device": "cpu",
                "local_files_only": True,
            },
            "data": {"emotions": emotions},
        },
    )
    write_json(bundle / "metrics.json", {"selected_layer": 1})
    write_json(
        bundle / "environment.json",
        {"created_at": "2026-06-29T12:00:00+00:00", "resolved_model_revision": "abc123"},
    )
    return bundle


def test_activity_pair_config_parsing_builds_prompt():
    pairs = parse_activity_pairs(
        [
            {
                "id": "social-vs-rest",
                "activity_a": "call a friend",
                "category_a": "social_connection",
                "activity_b": "sit quietly alone",
                "category_b": "rest_withdrawal",
            }
        ]
    )

    assert pairs[0]["id"] == "social-vs-rest"
    assert "A: call a friend" in pairs[0]["prompt"]
    assert pairs[0]["prompt"].endswith("The better next action is:")


def test_choice_token_ids_and_margin_resolution():
    tokenizer = FakeTokenizer()
    ids = choice_token_ids(tokenizer)
    logits = torch.zeros(1, 32)
    logits[0, 20] = 3.0
    logits[0, 21] = 1.0

    assert ids == {"A": [20], "B": [21]}
    assert float(preference_margin(logits, ids)[0]) == pytest.approx(2.0)


def test_expected_direction_uses_emotion_category_mapping():
    mapping = {"happy": ["social_connection"], "sad": ["rest_withdrawal"]}

    assert expected_direction("happy", "social_connection", "rest_withdrawal", mapping) == 1
    assert expected_direction("sad", "social_connection", "rest_withdrawal", mapping) == -1
    assert expected_direction("angry", "social_connection", "rest_withdrawal", mapping) == 0


def test_aggregate_preference_rows_tracks_expected_effect():
    rows = [
        {
            "target_emotion": "happy",
            "control": "real",
            "raw_strength": 1.0,
            "margin_delta_from_baseline": 2.0,
            "expected_effect": 2.0,
            "matches_hypothesis": True,
            "kl_from_baseline": 0.1,
            "max_abs_logit_change": 3.0,
        },
        {
            "target_emotion": "happy",
            "control": "real",
            "raw_strength": 1.0,
            "margin_delta_from_baseline": -1.0,
            "expected_effect": 1.0,
            "matches_hypothesis": True,
            "kl_from_baseline": 0.3,
            "max_abs_logit_change": 1.0,
        },
    ]

    aggregate = aggregate_preference_rows(rows)["rows"][0]

    assert aggregate["mean_margin_delta"] == pytest.approx(0.5)
    assert aggregate["mean_expected_effect"] == pytest.approx(1.5)
    assert aggregate["mean_kl_from_baseline"] == pytest.approx(0.2)


def test_zero_fidelity_reads_zero_strength_rows():
    scores = {
        "rows": [
            {"raw_strength": -1.0, "max_abs_logit_change": 2.0},
            {"raw_strength": 0.0, "max_abs_logit_change": 0.0},
            {"raw_strength": 0.0, "max_abs_logit_change": 0.25},
        ]
    }

    assert zero_fidelity(scores) == {"max_abs_logit_error": 0.25, "rows": 2}


def test_prototype5_run_writes_preference_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(prototype5.AutoTokenizer, "from_pretrained", FakeTokenizer.from_pretrained)
    monkeypatch.setattr(
        prototype5.AutoModelForCausalLM, "from_pretrained", FakeModel.from_pretrained
    )
    bundle = make_bundle(tmp_path)
    config = {
        "experiment": "prototype5-test",
        "seed": 42,
        "prototype1": {
            "run_dir": str(bundle),
            "emotion_vectors": "emotion_vectors.safetensors",
        },
        "model": {"device": "cpu", "dtype": "float32", "local_files_only": True},
        "emotions": ["happy", "sad", "angry", "afraid"],
        "intervention": {
            "layer": None,
            "strengths": [-1, 0, 1],
            "scale_by_vector_norm": False,
        },
        "preference_task": {
            "expected_categories": {
                "happy": ["social_connection"],
                "sad": ["rest_withdrawal"],
                "angry": ["confrontation_correction"],
                "afraid": ["safety_checking"],
            },
            "pairs": [
                {
                    "id": "happy-pair",
                    "activity_a": "call a friend",
                    "category_a": "social_connection",
                    "activity_b": "start cooking dinner",
                    "category_b": "routine_maintenance",
                },
                {
                    "id": "sad-pair",
                    "activity_a": "sit quietly alone",
                    "category_a": "rest_withdrawal",
                    "activity_b": "start cooking dinner",
                    "category_b": "routine_maintenance",
                },
                {
                    "id": "angry-pair",
                    "activity_a": "confront the person immediately",
                    "category_a": "confrontation_correction",
                    "activity_b": "start cooking dinner",
                    "category_b": "routine_maintenance",
                },
                {
                    "id": "afraid-pair",
                    "activity_a": "check that the door is locked",
                    "category_a": "safety_checking",
                    "activity_b": "start cooking dinner",
                    "category_b": "routine_maintenance",
                },
            ],
        },
        "controls": {"random_seed": 99},
        "gates": {"zero_steering_max_abs_logit_error": 1e-6},
        "output_dir": str(tmp_path / "prototype5"),
    }

    output = run(config)

    metrics = json.loads((output / "metrics.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    scores = json.loads((output / "diagnostics" / "preference_scores.json").read_text())
    controls = json.loads((output / "diagnostics" / "controls.json").read_text())
    assert metrics["all_hard_gates_pass"]
    assert metrics["hard_gates"]["random_vector_control_recorded"]
    assert metrics["hard_gates"]["wrong_emotion_control_recorded"]
    assert metrics["hard_gates"]["pairwise_preference_scores_recorded"]
    assert metrics["zero_fidelity"]["max_abs_logit_error"] == 0.0
    assert summary["interpretation_caveat"] == INTERPRETATION_CAVEAT
    assert {row["control"] for row in scores["rows"]} == {
        "real",
        "random",
        "wrong_emotion",
    }
    assert controls["random_vector"]["matched_norm"] is True
    assert (output / "diagnostics" / "kl.json").is_file()
    assert (output / "diagnostics" / "elo.json").is_file()


def test_prototype5_config_points_to_canonical_bundle():
    config = yaml.safe_load(Path("configs/prototype5.yaml").read_text())

    assert config["experiment"] == "prototype5"
    assert "colab-run-prototype-2.5" in config["prototype1"]["run_dir"]
    assert config["prototype1"]["emotion_vectors"] == "emotion_vectors.safetensors"
    assert 0 in config["intervention"]["strengths"]
    assert config["preference_task"]["expected_categories"]["happy"]
    assert len(config["preference_task"]["pairs"]) >= 8
    assert config["output_dir"] == "artifacts/prototype5"

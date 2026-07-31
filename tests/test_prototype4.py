import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml
from safetensors.torch import save_file

from functional_emotions import prototype4
from functional_emotions.prototype4 import (
    INTERPRETATION_CAVEAT,
    aggregate_matching_rows,
    lexical_emotion_scores,
    matched_random_vector,
    resolve_selected_layer,
    run,
    token_ids_for_terms,
    wrong_emotion,
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
            "happy": 10,
            "sad": 11,
            "angry": 12,
            "afraid": 13,
            "joyful": 14,
            "sorrowful": 15,
            "furious": 16,
            "scared": 17,
        }
        self.inverse = {value: key for key, value in self.vocab.items()}

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
            return {"input_ids": self.encode(texts)}
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

    def batch_decode(self, sequences, skip_special_tokens=True):
        texts = []
        for sequence in sequences.tolist():
            tokens = []
            for token_id in sequence:
                if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                    continue
                tokens.append(self.inverse.get(int(token_id), "neutral"))
            texts.append(" ".join(tokens))
        return texts


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
            self.lm_head.weight[10, 0] = 1.0
            self.lm_head.weight[11, 1] = 1.0
            self.lm_head.weight[12, 2] = 1.0
            self.lm_head.weight[13, 3] = 1.0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def forward(self, input_ids, attention_mask=None):
        _ = attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **_kwargs):
        generated = input_ids
        mask = attention_mask
        for _ in range(max_new_tokens):
            output = self(input_ids=generated, attention_mask=mask)
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if mask is not None:
                mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)
        return generated


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


def test_token_ids_include_space_prefixed_variants():
    tokenizer = FakeTokenizer()

    assert token_ids_for_terms(tokenizer, ["happy"]) == [10]


def test_random_vector_control_matches_norm():
    vector = torch.tensor([3.0, 4.0])

    control = matched_random_vector(vector, seed=123)

    assert control.norm() == pytest.approx(vector.norm())
    assert not torch.allclose(control, vector)


def test_wrong_emotion_rotates_to_a_different_label():
    assert wrong_emotion("happy", ["happy", "sad", "angry"]) == "sad"
    assert wrong_emotion("angry", ["happy", "sad", "angry"]) == "happy"


def test_selected_layer_uses_metrics_then_layer_19_fallback():
    config = {"intervention": {"layer": None}}

    assert resolve_selected_layer(config, {"selected_layer": 7}) == 7
    assert resolve_selected_layer(config, {}) == 19
    assert resolve_selected_layer({"intervention": {"layer": 3}}, {"selected_layer": 7}) == 3


def test_aggregate_matching_rows_tracks_specificity():
    rows = [
        {"target_emotion": "happy", "control": "real", "raw_strength": 1.0, "target_delta": 2.0, "specificity_delta": 1.5, "kl_from_baseline": 0.1, "max_abs_logit_change": 2.0},
        {"target_emotion": "happy", "control": "real", "raw_strength": 1.0, "target_delta": 1.0, "specificity_delta": 0.5, "kl_from_baseline": 0.3, "max_abs_logit_change": 1.0},
    ]

    aggregate = aggregate_matching_rows(rows)["rows"][0]

    assert aggregate["mean_target_delta"] == pytest.approx(1.5)
    assert aggregate["mean_specificity_delta"] == pytest.approx(1.0)
    assert aggregate["mean_kl_from_baseline"] == pytest.approx(0.2)


def test_lexical_emotion_scores_count_target_specific_terms():
    terms = {
        "happy": ["happy", "smiled"],
        "sad": ["sad", "tears"],
    }

    scores = lexical_emotion_scores("Avery smiled and felt happy.", terms, "happy")

    assert scores["emotion_term_counts"]["happy"] == 2
    assert scores["emotion_term_counts"]["sad"] == 0
    assert scores["target_term_count"] == 2
    assert scores["lexical_specificity"] == pytest.approx(2.0)


def test_prototype4_run_writes_steering_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(prototype4.AutoTokenizer, "from_pretrained", FakeTokenizer.from_pretrained)
    monkeypatch.setattr(
        prototype4.AutoModelForCausalLM, "from_pretrained", FakeModel.from_pretrained
    )
    bundle = make_bundle(tmp_path)
    config = {
        "experiment": "prototype4-test",
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
        "matching_token": {
            "terms": {
                "happy": ["happy"],
                "sad": ["sad"],
                "angry": ["angry"],
                "afraid": ["afraid"],
            },
            "prompts": [
                {
                    "id": f"mt-{emotion}",
                    "target_emotion": emotion,
                    "text": "neutral",
                }
                for emotion in ["happy", "sad", "angry", "afraid"]
            ],
        },
        "free_generation": {
            "strengths": [0, 1],
            "max_new_tokens": 2,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "samples_per_condition": 2,
            "seed": 1234,
            "terms": {
                "happy": ["happy"],
                "sad": ["sad"],
                "angry": ["angry"],
                "afraid": ["afraid"],
            },
            "prompts": [
                {"id": "fg-happy", "target_emotion": "happy", "text": "neutral"}
            ],
        },
        "controls": {"random_seed": 99},
        "gates": {"zero_steering_max_abs_logit_error": 1e-6},
        "output_dir": str(tmp_path / "prototype4"),
    }

    output = run(config)

    metrics = json.loads((output / "metrics.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    matching = json.loads(
        (output / "diagnostics" / "matching_token_scores.json").read_text()
    )
    assert metrics["all_hard_gates_pass"]
    assert metrics["hard_gates"]["random_vector_control_recorded"]
    assert metrics["zero_fidelity"]["max_abs_logit_error"] == 0.0
    assert metrics["matching_token"]["positive_strength_summary"]["mean_target_delta"] > 0.0
    assert metrics["free_generation"]["sample_count"] == 4
    assert metrics["free_generation"]["mean_positive_target_term_count"] > 0.0
    assert summary["interpretation_caveat"] == INTERPRETATION_CAVEAT
    statistics = metrics["statistics"]
    assert statistics["target_delta"]["bootstrap"]["n_clusters"] == 4
    assert "adjudication" in statistics
    assert "target_delta_ci_low" in summary
    assert "specificity_minus_target_ci_excludes_zero" in summary
    free_generation = json.loads(
        (output / "diagnostics" / "free_generation_samples.json").read_text()
    )
    assert free_generation["generation_config"]["do_sample"] is True
    assert free_generation["generation_config"]["samples_per_condition"] == 2
    assert all("emotion_term_counts" in sample for sample in free_generation["samples"])
    assert (output / "diagnostics" / "kl_fluency.json").is_file()
    assert (output / "diagnostics" / "controls.json").is_file()
    assert {row["control"] for row in matching["rows"]} == {
        "real",
        "random",
        "wrong_emotion",
    }


def test_prototype4_config_points_to_canonical_bundle():
    config = yaml.safe_load(Path("configs/prototype4.yaml").read_text())

    assert config["experiment"] == "prototype4"
    assert "colab-run-prototype-2.5" in config["prototype1"]["run_dir"]
    assert config["prototype1"]["emotion_vectors"] == "emotion_vectors.safetensors"
    assert 0 in config["intervention"]["strengths"]
    assert config["free_generation"]["do_sample"] is True
    assert config["free_generation"]["samples_per_condition"] > 1
    assert max(config["free_generation"]["strengths"]) >= 8
    assert config["output_dir"] == "artifacts/prototype4"

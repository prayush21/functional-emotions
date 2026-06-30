import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml
from safetensors.torch import save_file

from functional_emotions import prototype51
from functional_emotions.prototype51 import (
    CHOICE_TOKEN_MARGIN,
    INTERPRETATION_CAVEAT,
    OPTION_TEXT_LOGPROB_MARGIN,
    aggregate_rows,
    build_preference_metrics,
    choice_margin,
    choice_token_ids,
    expected_direction,
    option_text_margin,
    order_swapped_pairs,
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
            "unk": 2,
            "the": 3,
            "person": 4,
            "is": 5,
            "deciding": 6,
            "what": 7,
            "to": 8,
            "do": 9,
            "next": 10,
            "better": 11,
            "action": 12,
            "A": 20,
            "B": 21,
            "call": 22,
            "a": 23,
            "friend": 24,
            "sit": 25,
            "quietly": 26,
            "alone": 27,
            "check": 28,
            "door": 29,
            "start": 30,
            "cooking": 31,
            "dinner": 32,
            "confront": 33,
            "rest": 34,
        }

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        tokens = re.findall(r"[A-Za-z]+", text)
        ids = []
        for token in tokens:
            key = token if token in {"A", "B"} else token.lower()
            ids.append(self.vocab.get(key, self.vocab["unk"]))
        return ids or [self.vocab["unk"]]

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
        self.embedding = nn.Embedding(64, 4)
        self.lm_head = nn.Linear(4, 64, bias=False)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.lm_head.weight.zero_()
            for token in (20, 22, 24, 28, 29):
                self.lm_head.weight[token, 0] = 1.0
            for token in (21, 25, 26, 27, 30, 31, 32):
                self.lm_head.weight[token, 0] = -1.0

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


def tiny_config(tmp_path: Path, bundle: Path) -> dict:
    return {
        "experiment": "prototype51-test",
        "seed": 42,
        "prototype1": {
            "run_dir": str(bundle),
            "emotion_vectors": "emotion_vectors.safetensors",
        },
        "model": {"device": "cpu", "dtype": "float32", "local_files_only": True},
        "emotions": ["happy", "sad", "angry", "afraid"],
        "intervention": {
            "layers": [0, 1],
            "layer": None,
            "strengths": [-1, 0, 1],
            "scale_by_vector_norm": False,
        },
        "scoring": {
            "primary": OPTION_TEXT_LOGPROB_MARGIN,
            "modes": [CHOICE_TOKEN_MARGIN, OPTION_TEXT_LOGPROB_MARGIN],
        },
        "preference_task": {
            "contexts": [
                {
                    "id": "neutral",
                    "family": "neutral",
                    "text": "The person is deciding what to do next.",
                }
            ],
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
                    "activity_b": "sit quietly alone",
                    "category_b": "rest_withdrawal",
                }
            ],
        },
        "controls": {"random_seed": 99},
        "gates": {
            "zero_steering_max_abs_logit_error": 1e-6,
            "zero_steering_max_abs_logprob_error": 1e-6,
            "kl_max_for_effect_summary": 10.0,
        },
        "output_dir": str(tmp_path / "prototype51"),
    }


def test_order_swap_pair_generation_is_symmetric():
    pairs = order_swapped_pairs(
        [
            {
                "id": "p1",
                "activity_a": "call a friend",
                "category_a": "social_connection",
                "activity_b": "sit quietly alone",
                "category_b": "rest_withdrawal",
            }
        ]
    )

    assert [pair["order"] for pair in pairs] == ["original", "swapped"]
    assert pairs[1]["activity_a"] == "sit quietly alone"
    assert pairs[1]["category_b"] == "social_connection"


def test_option_text_logprob_margin_calculation_uses_full_text():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    layer = model.model.layers[0]
    prefix = "The better next action is"

    baseline = option_text_margin(
        model=model,
        layer=layer,
        tokenizer=tokenizer,
        prefix=prefix,
        activity_a="call a friend",
        activity_b="sit quietly alone",
        scoring_mode=OPTION_TEXT_LOGPROB_MARGIN,
        device="cpu",
    )
    steered = option_text_margin(
        model=model,
        layer=layer,
        tokenizer=tokenizer,
        prefix=prefix,
        activity_a="call a friend",
        activity_b="sit quietly alone",
        scoring_mode=OPTION_TEXT_LOGPROB_MARGIN,
        device="cpu",
        vector=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        strength=1.0,
    )

    assert steered["margin_a_minus_b"] > baseline["margin_a_minus_b"]


def test_choice_token_margin_still_works():
    tokenizer = FakeTokenizer()
    ids = choice_token_ids(tokenizer)
    logits = torch.zeros(1, 64)
    logits[0, 20] = 3.0
    logits[0, 21] = 1.0

    assert ids == {"A": [20], "B": [21]}
    assert float(choice_margin(logits, ids)[0]) == pytest.approx(2.0)


def test_expected_direction_aggregation_is_order_invariant():
    mapping = {"happy": ["social_connection"]}

    assert expected_direction("happy", "social_connection", "rest_withdrawal", mapping) == 1
    assert expected_direction("happy", "rest_withdrawal", "social_connection", mapping) == -1


def test_per_emotion_and_per_layer_aggregation():
    rows = [
        {
            "vector_layer": 0,
            "scoring_mode": OPTION_TEXT_LOGPROB_MARGIN,
            "target_emotion": "happy",
            "control": "real",
            "raw_strength": 1.0,
            "margin_delta_from_baseline": 1.0,
            "expected_effect": 1.0,
            "matches_hypothesis": True,
            "kl_from_baseline": 0.1,
            "max_abs_logit_change": 1.0,
            "max_abs_score_change": 1.0,
        },
        {
            "vector_layer": 1,
            "scoring_mode": OPTION_TEXT_LOGPROB_MARGIN,
            "target_emotion": "sad",
            "control": "real",
            "raw_strength": 1.0,
            "margin_delta_from_baseline": 2.0,
            "expected_effect": 2.0,
            "matches_hypothesis": True,
            "kl_from_baseline": 0.1,
            "max_abs_logit_change": 2.0,
            "max_abs_score_change": 2.0,
        },
    ]

    per_emotion = aggregate_rows(rows, ["target_emotion"])
    per_layer = aggregate_rows(rows, ["vector_layer"])

    assert {row["target_emotion"] for row in per_emotion} == {"happy", "sad"}
    assert {row["vector_layer"] for row in per_layer} == {0, 1}


def test_kl_guardrail_filtering_excludes_high_kl_rows():
    rows = []
    for kl, effect in [(0.1, 1.0), (0.8, 9.0)]:
        rows.append(
            {
                "vector_layer": 0,
                "model_layer": 0,
                "scoring_mode": OPTION_TEXT_LOGPROB_MARGIN,
                "context_id": "neutral",
                "context_family": "neutral",
                "context": "The person is deciding what to do next.",
                "pair_id": "p::original",
                "base_pair_id": "p",
                "order": "original",
                "prompt": "prompt",
                "activity_a": "call a friend",
                "activity_b": "sit quietly alone",
                "category_a": "social_connection",
                "category_b": "rest_withdrawal",
                "target_emotion": "happy",
                "vector_label": "happy",
                "control": "real",
                "raw_strength": 1.0,
                "applied_strength": 1.0,
                "baseline_score_a": 0.0,
                "baseline_score_b": 0.0,
                "score_a": effect,
                "score_b": 0.0,
                "baseline_margin_a_minus_b": 0.0,
                "margin_a_minus_b": effect,
                "margin_delta_from_baseline": effect,
                "expected_direction": 1,
                "expected_effect": effect,
                "matches_hypothesis": True,
                "max_abs_logit_change": effect,
                "max_abs_score_change": effect,
                "kl_from_baseline": kl,
            }
        )

    metrics = build_preference_metrics(rows, OPTION_TEXT_LOGPROB_MARGIN, kl_max=0.25)

    assert metrics["positive_strength_summary"]["mean_expected_effect"] == pytest.approx(1.0)


def test_zero_fidelity_reads_logits_and_logprob_errors():
    rows = [
        {"raw_strength": 0.0, "max_abs_logit_change": 0.0, "max_abs_score_change": 0.0},
        {"raw_strength": 0.0, "max_abs_logit_change": 0.1, "max_abs_score_change": 0.2},
        {"raw_strength": 1.0, "max_abs_logit_change": 9.0, "max_abs_score_change": 9.0},
    ]

    assert zero_fidelity(rows) == {
        "max_abs_logit_error": 0.1,
        "max_abs_logprob_or_margin_error": 0.2,
        "rows": 2,
    }


def test_prototype51_run_writes_robust_preference_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prototype51.AutoTokenizer, "from_pretrained", FakeTokenizer.from_pretrained
    )
    monkeypatch.setattr(
        prototype51.AutoModelForCausalLM, "from_pretrained", FakeModel.from_pretrained
    )
    bundle = make_bundle(tmp_path)

    output = run(tiny_config(tmp_path, bundle))

    metrics = json.loads((output / "metrics.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    scores = json.loads((output / "diagnostics" / "preference_scores.json").read_text())
    controls = json.loads((output / "diagnostics" / "controls.json").read_text())
    assert metrics["all_hard_gates_pass"]
    assert metrics["hard_gates"]["option_text_scoring_diagnostics_recorded"]
    assert metrics["hard_gates"]["order_swap_diagnostics_recorded"]
    assert metrics["hard_gates"]["random_vector_control_recorded"]
    assert metrics["hard_gates"]["wrong_emotion_control_recorded"]
    assert metrics["zero_fidelity"]["max_abs_logit_error"] == 0.0
    assert summary["interpretation_caveat"] == INTERPRETATION_CAVEAT
    assert metrics["evaluated_layers"] == [0, 1]
    assert {row["control"] for row in scores["rows"]} == {
        "real",
        "random",
        "wrong_emotion",
    }
    assert controls["random_vector"]["matched_norm"] is True
    assert (output / "diagnostics" / "order_swap_scores.json").is_file()
    assert (output / "diagnostics" / "option_text_scores.json").is_file()
    assert (output / "diagnostics" / "layer_sweep.json").is_file()
    assert (output / "diagnostics" / "kl.json").is_file()


def test_prototype51_config_points_to_canonical_bundle():
    config = yaml.safe_load(Path("configs/prototype51.yaml").read_text())

    assert config["experiment"] == "prototype51"
    assert "colab-run-prototype-2.5" in config["prototype1"]["run_dir"]
    assert config["prototype1"]["emotion_vectors"] == "emotion_vectors.safetensors"
    assert config["intervention"]["layers"] == [16, 17, 18, 19, 20, 21, 22]
    assert 0 in config["intervention"]["strengths"]
    assert config["scoring"]["primary"] == OPTION_TEXT_LOGPROB_MARGIN
    assert OPTION_TEXT_LOGPROB_MARGIN in config["scoring"]["modes"]
    assert len(config["preference_task"]["contexts"]) >= 5
    assert len(config["preference_task"]["pairs"]) >= 8
    assert config["gates"]["kl_max_for_effect_summary"] == 0.25
    assert config["output_dir"] == "artifacts/prototype51"

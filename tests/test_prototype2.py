import math

import pytest
import torch

from functional_emotions.prototype2 import (
    apply_zscore_calibration,
    calibration_parameters,
    confusion_diagnostics,
    default_lexical_scenarios,
    evaluate_score_matrix,
    logit_lens_diagnostic,
    metric_deltas,
    pca_comparison,
    shuffled_label_control,
    spearman,
    topic_stratified_metrics,
)


def test_score_matrix_metrics_match_argmax_and_margin():
    scores = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
    result = evaluate_score_matrix(scores, ["a", "a", "b", "b"], ["a", "b"])

    assert result["accuracy"] == 1.0
    assert result["macro_auc"] == 1.0
    assert result["mean_correct_margin"] == 1.5


def test_shuffled_label_control_reports_real_minus_control_effect():
    activations = torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]])
    vectors = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])

    rows = shuffled_label_control(activations, ["a", "b"], ["a", "b"], vectors, seed=1)

    assert rows[0]["real"]["accuracy"] == 1.0
    assert rows[0]["shuffled"]["accuracy"] == 0.0
    assert rows[0]["effect"]["accuracy"] == 1.0


def test_pca_comparison_uses_clean_minus_raw_deltas():
    raw = [{"accuracy": 0.25, "macro_auc": 0.5, "mean_correct_margin": -1.0}]
    clean = [{"accuracy": 0.5, "macro_auc": 0.75, "mean_correct_margin": -0.25}]

    rows = pca_comparison(raw, clean, [3])

    assert rows[0]["layer"] == 3
    assert rows[0]["pca_minus_raw"] == {
        "accuracy": 0.25,
        "macro_auc": 0.25,
        "mean_correct_margin": 0.75,
    }


def test_zscore_calibration_uses_train_score_scale():
    train_scores = torch.tensor([[2.0, 10.0], [4.0, 14.0], [6.0, 18.0]])
    params = calibration_parameters(train_scores)
    calibrated = apply_zscore_calibration(train_scores, params)

    assert torch.allclose(calibrated.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(calibrated.std(dim=0), torch.ones(2), atol=1e-6)


def test_confusion_diagnostics_reports_angry_dominance_and_afraid_wins():
    scores = [torch.tensor([[0.0, 0.0, 3.0, 1.0], [0.0, 0.0, 2.0, 4.0]])]
    validation = [
        {
            "confusion_matrix": {
                "labels": ["happy", "sad", "angry", "afraid"],
                "rows_are_true_labels": [[0, 0, 0, 0]],
            }
        }
    ]

    result = confusion_diagnostics(
        scores, validation, ["happy", "sad", "angry", "afraid"], [7]
    )

    assert result["afraid_ever_wins_argmax"] is True
    assert result["angry_is_top_win_rate_layers"] == [7]
    assert result["layers"][0]["win_rates"]["angry"] == 0.5


def test_spearman_handles_monotonic_and_constant_inputs():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 2, 3], [5, 5, 5]))


def test_metric_deltas_are_real_minus_control():
    assert metric_deltas(
        {"accuracy": 0.6, "macro_auc": 0.8, "mean_correct_margin": 0.1},
        {"accuracy": 0.2, "macro_auc": 0.5, "mean_correct_margin": -0.4},
    ) == {
        "accuracy": 0.39999999999999997,
        "macro_auc": 0.30000000000000004,
        "mean_correct_margin": 0.5,
    }


def test_default_lexical_controls_cover_each_emotion():
    rows = default_lexical_scenarios()
    controls = {
        row["emotion"]
        for row in rows
        if row.get("variant") == "minimal_control"
    }

    assert controls == {"happy", "sad", "angry", "afraid"}


def test_topic_stratified_metrics_reports_topic_minimums():
    rows = [
        {"topic": "one", "emotion": "a"},
        {"topic": "one", "emotion": "b"},
        {"topic": "two", "emotion": "a"},
        {"topic": "two", "emotion": "b"},
    ]
    scores = [torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])]

    result = topic_stratified_metrics(rows, scores, ["a", "b"], [5])

    assert result["layers"][0]["minimum_topic_accuracy"] == 0.5
    assert result["layers"][0]["mean_topic_accuracy"] == 0.75
    assert [row["topic"] for row in result["layers"][0]["topics"]] == ["one", "two"]


def test_logit_lens_diagnostic_scores_emotion_tokens_with_fake_head():
    class TinyTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0 if text.strip() == "a" else 1]}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

        def get_output_embeddings(self):
            return self.head

    activations = torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]])

    result = logit_lens_diagnostic(
        TinyModel(),
        TinyTokenizer(),
        activations,
        ["a", "b"],
        ["a", "b"],
        [0],
        terms_by_emotion={"a": ["a"], "b": ["b"]},
    )

    assert result["available"] is True
    assert result["layers"][0]["metrics"]["accuracy"] == 1.0
    assert result["emotion_token_ids"] == {"a": [0], "b": [1]}

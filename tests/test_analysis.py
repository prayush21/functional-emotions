import json
import math
from pathlib import Path

import pytest

from functional_emotions.analysis import (
    analyze,
    analyze_run_dir,
    detect_kind,
    prototype4_statistics,
    prototype51_statistics,
    statistics_params,
)


def p4_row(prompt_id, control, strength, target_delta, non_target_mean):
    return {
        "prompt_id": prompt_id,
        "target_emotion": "happy",
        "control": control,
        "raw_strength": strength,
        "target_delta": target_delta,
        "non_target_mean_delta": non_target_mean,
        "specificity_delta": target_delta - non_target_mean,
        "kl_from_baseline": 0.01,
    }


def make_p4_rows():
    rows = []
    for index, prompt_id in enumerate(["p1", "p2", "p3", "p4"]):
        for strength in (-1.0, 0.0, 1.0, 2.0):
            rows.append(
                p4_row(prompt_id, "real", strength, 0.1 * strength + 0.01 * index, -0.02)
            )
            rows.append(p4_row(prompt_id, "random", strength, 0.001, 0.0))
            rows.append(p4_row(prompt_id, "wrong_emotion", strength, -0.05, 0.01))
    return rows


def p51_row(
    *,
    pair,
    context="neutral",
    control="real",
    strength=1.0,
    effect=0.3,
    kl=0.05,
    scoring_mode="option_text_logprob_margin",
    layer=1,
):
    return {
        "vector_layer": layer,
        "scoring_mode": scoring_mode,
        "context_id": context,
        "context_family": context,
        "pair_id": f"{pair}::original",
        "base_pair_id": pair,
        "order": "original",
        "target_emotion": "happy",
        "control": control,
        "raw_strength": strength,
        "margin_delta_from_baseline": effect,
        "expected_direction": 1,
        "expected_effect": effect,
        "matches_hypothesis": True,
        "max_abs_logit_change": abs(effect),
        "max_abs_score_change": abs(effect),
        "kl_from_baseline": kl,
    }


def make_p51_rows():
    rows = []
    for pair in ("pair-a", "pair-b", "pair-c"):
        for context in ("neutral", "conflict"):
            for strength in (1.0, 2.0):
                rows.append(
                    p51_row(pair=pair, context=context, strength=strength, effect=0.3)
                )
                rows.append(
                    p51_row(
                        pair=pair,
                        context=context,
                        strength=strength,
                        control="random",
                        effect=0.01,
                    )
                )
                rows.append(
                    p51_row(
                        pair=pair,
                        context=context,
                        strength=strength,
                        control="wrong_emotion",
                        effect=-0.1,
                    )
                )
    # High-KL row that the guardrail must exclude from every estimate.
    rows.append(p51_row(pair="pair-a", strength=1.0, effect=99.0, kl=5.0))
    # Off-primary scoring mode row, also excluded.
    rows.append(
        p51_row(pair="pair-a", strength=1.0, effect=88.0, scoring_mode="choice_token_margin")
    )
    return rows


def test_prototype4_statistics_reports_cis_and_adjudication():
    report = prototype4_statistics(make_p4_rows(), n_resamples=200, seed=0)

    target = report["target_delta"]["bootstrap"]
    assert target["n_clusters"] == 4
    assert target["ci_low"] <= target["estimate"] <= target["ci_high"]
    adjudication = report["adjudication"]
    # non_target_mean_delta is -0.02 on every real row, so the paired
    # difference is exactly +0.02 with no variance across clusters.
    assert adjudication["mean_difference"] == pytest.approx(0.02)
    assert report["specificity_minus_target"]["sign_flip"]["method"] == "exact"
    assert report["control_target_delta"]["random"]["estimate"] == pytest.approx(0.001)


def test_prototype51_statistics_applies_kl_guardrail_and_pairs_controls():
    report = prototype51_statistics(
        make_p51_rows(),
        primary_scoring_mode="option_text_logprob_margin",
        kl_max=0.25,
        n_resamples=200,
        seed=0,
    )

    effect = report["expected_effect"]["bootstrap_by_pair"]
    assert effect["estimate"] == pytest.approx(0.3)
    assert effect["n_clusters"] == 3
    random_contrast = report["control_contrasts"]["random"]["real_minus_control"]
    assert random_contrast["mean_difference"]["estimate"] == pytest.approx(0.29)
    wrong_contrast = report["control_contrasts"]["wrong_emotion"]["real_minus_control"]
    assert wrong_contrast["mean_difference"]["estimate"] == pytest.approx(0.4)
    assert report["per_emotion_expected_effect"]["happy"]["estimate"] == pytest.approx(0.3)


def write_bundle(tmp_path: Path, name: str, kind: str, rows, seed=42) -> Path:
    run_dir = tmp_path / name
    (run_dir / "diagnostics").mkdir(parents=True)
    config = {
        "experiment": f"{kind}-test",
        "seed": seed,
        "scoring": {"primary": "option_text_logprob_margin"},
        "gates": {"kl_max_for_effect_summary": 0.25},
    }
    (run_dir / "config.json").write_text(json.dumps(config))
    filename = (
        "preference_scores.json" if kind == "prototype51" else "matching_token_scores.json"
    )
    (run_dir / "diagnostics" / filename).write_text(json.dumps({"rows": rows}))
    return run_dir


def test_detect_kind_and_analyze_run_dir(tmp_path):
    p4_dir = write_bundle(tmp_path, "p4-run", "prototype4", make_p4_rows())
    p51_dir = write_bundle(tmp_path, "p51-run", "prototype51", make_p51_rows())

    assert detect_kind(p4_dir) == "prototype4"
    assert detect_kind(p51_dir) == "prototype51"
    p4_report = analyze_run_dir(p4_dir, n_resamples=100)
    p51_report = analyze_run_dir(p51_dir, n_resamples=100)
    assert p4_report["kind"] == "prototype4"
    assert p51_report["headline_estimate"] == pytest.approx(0.3)


def test_detect_kind_rejects_unknown_bundle(tmp_path):
    empty = tmp_path / "empty"
    (empty / "diagnostics").mkdir(parents=True)

    with pytest.raises(ValueError):
        detect_kind(empty)


def test_analyze_multi_seed_summary(tmp_path):
    first = write_bundle(tmp_path, "seed42", "prototype51", make_p51_rows(), seed=42)
    second = write_bundle(tmp_path, "seed43", "prototype51", make_p51_rows(), seed=43)

    report = analyze([first, second], n_resamples=100)

    assert report["multi_seed"]["n_seeds"] == 2
    assert report["multi_seed"]["mean"] == pytest.approx(0.3)
    assert report["multi_seed"]["all_same_sign"] is True


def test_analyze_rejects_mixed_prototypes(tmp_path):
    p4_dir = write_bundle(tmp_path, "p4-run", "prototype4", make_p4_rows())
    p51_dir = write_bundle(tmp_path, "p51-run", "prototype51", make_p51_rows())

    with pytest.raises(ValueError):
        analyze([p4_dir, p51_dir], n_resamples=50)


def test_cli_writes_stats_summary(tmp_path, monkeypatch, capsys):
    from functional_emotions import analysis

    run_dir = write_bundle(tmp_path, "p51-run", "prototype51", make_p51_rows())
    monkeypatch.setattr(
        "sys.argv",
        [
            "fe-analyze-stats",
            "--run-dir",
            str(run_dir),
            "--n-resamples",
            "100",
            "--write",
        ],
    )

    analysis.main()

    written = json.loads((run_dir / "stats_summary.json").read_text())
    assert written["kind"] == "prototype51"
    printed = json.loads(capsys.readouterr().out)
    assert "disclaimer" in printed


def test_statistics_params_defaults_and_overrides():
    assert statistics_params({}) == {
        "n_resamples": 2000,
        "n_permutations": 10000,
        "seed": 0,
    }
    assert statistics_params({"statistics": {"n_resamples": 50, "seed": 9}}) == {
        "n_resamples": 50,
        "n_permutations": 10000,
        "seed": 9,
    }


def test_prototype4_statistics_handles_empty_rows():
    report = prototype4_statistics([], n_resamples=50)

    assert math.isnan(report["target_delta"]["bootstrap"]["estimate"])
    assert report["adjudication"]["ci_excludes_zero"] is False

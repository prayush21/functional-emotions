"""Statistics reporting for completed prototype runs (``fe-analyze-stats``).

Computes cluster-bootstrap confidence intervals and permutation nulls for the
Prototype 4 and Prototype 5.1 effect metrics, from the per-row diagnostics a
run bundle already contains. This is additional reporting: it never reads or
edits pre-registered hard gates, and the same helpers are called by the
runners to embed a ``statistics`` block in future runs' metrics.

Multi-seed usage: pass several ``--run-dir`` bundles of the same prototype and
the report adds an across-seed summary of the headline effect.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .stats import (
    DEFAULT_PERMUTATIONS,
    DEFAULT_RESAMPLES,
    bootstrap_mean_ci,
    multi_seed_summary,
    paired_difference_stats,
    sign_flip_pvalue,
)

STATISTICS_DISCLAIMER = (
    "Additional reporting only: these intervals and permutation p-values are not "
    "pre-registered hard gates and do not change any gate outcome. Findings remain "
    "functional/causal observations about model internals and behavior, not "
    "evidence of subjective experience."
)


def statistics_params(config: dict[str, Any]) -> dict[str, int]:
    block = config.get("statistics") or {}
    return {
        "n_resamples": int(block.get("n_resamples", DEFAULT_RESAMPLES)),
        "n_permutations": int(block.get("n_permutations", DEFAULT_PERMUTATIONS)),
        "seed": int(block.get("seed", 0)),
    }


def _real_positive(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows if row["control"] == "real" and float(row["raw_strength"]) > 0.0
    ]


def prototype4_statistics(
    rows: list[dict[str, Any]],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """CIs and permutation nulls for the matching-token effect rows.

    Clusters are prompts (``prompt_id``): every steering condition sharing a
    prompt is resampled together. Includes the specificity-versus-target
    adjudication: ``specificity_delta - target_delta`` equals
    ``-non_target_mean_delta`` per row, so a positive difference means
    non-target emotion logits moved down under steering.
    """

    real = _real_positive(rows)
    clusters = [row["prompt_id"] for row in real]
    target = [float(row["target_delta"]) for row in real]
    specificity = [float(row["specificity_delta"]) for row in real]
    kwargs = {"n_resamples": n_resamples, "confidence": 0.95, "seed": seed}
    report: dict[str, Any] = {
        "disclaimer": STATISTICS_DISCLAIMER,
        "cluster_level": "prompt_id",
        "selection": "control == real and raw_strength > 0",
        "target_delta": {
            "bootstrap": bootstrap_mean_ci(target, cluster_ids=clusters, **kwargs),
            "sign_flip": sign_flip_pvalue(
                target, cluster_ids=clusters, n_permutations=n_permutations, seed=seed
            ),
        },
        "specificity_delta": {
            "bootstrap": bootstrap_mean_ci(specificity, cluster_ids=clusters, **kwargs),
            "sign_flip": sign_flip_pvalue(
                specificity, cluster_ids=clusters, n_permutations=n_permutations, seed=seed
            ),
        },
        "specificity_minus_target": paired_difference_stats(
            specificity,
            target,
            cluster_ids=clusters,
            n_resamples=n_resamples,
            n_permutations=n_permutations,
            seed=seed,
        ),
    }
    controls: dict[str, Any] = {}
    for control in ("random", "wrong_emotion"):
        control_rows = [
            row
            for row in rows
            if row["control"] == control and float(row["raw_strength"]) > 0.0
        ]
        controls[control] = bootstrap_mean_ci(
            [float(row["target_delta"]) for row in control_rows],
            cluster_ids=[row["prompt_id"] for row in control_rows],
            **kwargs,
        )
    report["control_target_delta"] = controls
    difference = report["specificity_minus_target"]["mean_difference"]
    report["adjudication"] = {
        "question": (
            "Is specificity_delta > target_delta (equivalently, do non-target "
            "emotion logits move down under real positive steering)?"
        ),
        "mean_difference": difference["estimate"],
        "ci_low": difference["ci_low"],
        "ci_high": difference["ci_high"],
        "ci_excludes_zero": bool(
            not math.isnan(difference["ci_low"]) and difference["ci_low"] > 0.0
        )
        or bool(not math.isnan(difference["ci_high"]) and difference["ci_high"] < 0.0),
        "sign_flip_p_value": report["specificity_minus_target"]["sign_flip"]["p_value"],
    }
    return report


def _eligible_p51_rows(
    rows: list[dict[str, Any]],
    *,
    control: str,
    primary_scoring_mode: str,
    kl_max: float | None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["scoring_mode"] == primary_scoring_mode
        and row["control"] == control
        and float(row["raw_strength"]) > 0.0
        and row.get("matches_hypothesis")
        and math.isfinite(float(row["expected_effect"]))
        and (kl_max is None or float(row["kl_from_baseline"]) <= kl_max)
    ]


def prototype51_statistics(
    rows: list[dict[str, Any]],
    *,
    primary_scoring_mode: str,
    kl_max: float | None,
    n_resamples: int = DEFAULT_RESAMPLES,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """CIs and permutation nulls for the KL-filtered expected preference effect.

    The primary cluster level is the base activity pair (``base_pair_id``),
    the unit whose contents are correlated across contexts, orders, layers,
    and strengths; a context-level clustering is reported as a sensitivity
    check. Real-versus-control contrasts are paired within identical
    layer/context/pair/emotion/strength conditions.
    """

    real = _eligible_p51_rows(
        rows, control="real", primary_scoring_mode=primary_scoring_mode, kl_max=kl_max
    )
    effects = [float(row["expected_effect"]) for row in real]
    pair_clusters = [row["base_pair_id"] for row in real]
    context_clusters = [row["context_id"] for row in real]
    kwargs = {"n_resamples": n_resamples, "confidence": 0.95, "seed": seed}
    report: dict[str, Any] = {
        "disclaimer": STATISTICS_DISCLAIMER,
        "primary_scoring_mode": primary_scoring_mode,
        "kl_max": kl_max,
        "cluster_level": "base_pair_id (sensitivity: context_id)",
        "selection": (
            "primary scoring mode, control == real, raw_strength > 0, "
            "matches_hypothesis, KL <= kl_max"
        ),
        "expected_effect": {
            "bootstrap_by_pair": bootstrap_mean_ci(effects, cluster_ids=pair_clusters, **kwargs),
            "bootstrap_by_context": bootstrap_mean_ci(
                effects, cluster_ids=context_clusters, **kwargs
            ),
            "sign_flip_by_pair": sign_flip_pvalue(
                effects, cluster_ids=pair_clusters, n_permutations=n_permutations, seed=seed
            ),
        },
    }
    per_emotion: dict[str, Any] = {}
    for emotion in sorted({row["target_emotion"] for row in real}):
        emotion_rows = [row for row in real if row["target_emotion"] == emotion]
        per_emotion[emotion] = bootstrap_mean_ci(
            [float(row["expected_effect"]) for row in emotion_rows],
            cluster_ids=[row["base_pair_id"] for row in emotion_rows],
            **kwargs,
        )
    report["per_emotion_expected_effect"] = per_emotion

    def condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row["vector_layer"],
            row["context_id"],
            row["pair_id"],
            row["target_emotion"],
            float(row["raw_strength"]),
        )

    real_by_key = {condition_key(row): float(row["expected_effect"]) for row in real}
    contrasts: dict[str, Any] = {}
    for control in ("random", "wrong_emotion"):
        control_rows = _eligible_p51_rows(
            rows, control=control, primary_scoring_mode=primary_scoring_mode, kl_max=kl_max
        )
        control_effects = [float(row["expected_effect"]) for row in control_rows]
        entry: dict[str, Any] = {
            "expected_effect": bootstrap_mean_ci(
                control_effects,
                cluster_ids=[row["base_pair_id"] for row in control_rows],
                **kwargs,
            )
        }
        paired_real = []
        paired_control = []
        paired_clusters = []
        for row in control_rows:
            key = condition_key(row)
            if key in real_by_key:
                paired_real.append(real_by_key[key])
                paired_control.append(float(row["expected_effect"]))
                paired_clusters.append(row["base_pair_id"])
        entry["real_minus_control"] = paired_difference_stats(
            paired_real,
            paired_control,
            cluster_ids=paired_clusters,
            n_resamples=n_resamples,
            n_permutations=n_permutations,
            seed=seed,
        )
        contrasts[control] = entry
    report["control_contrasts"] = contrasts
    return report


def detect_kind(run_dir: Path) -> str:
    if (run_dir / "diagnostics" / "preference_scores.json").is_file():
        return "prototype51"
    if (run_dir / "diagnostics" / "matching_token_scores.json").is_file():
        return "prototype4"
    raise ValueError(
        f"{run_dir} has neither preference_scores.json nor matching_token_scores.json; "
        "fe-analyze-stats supports Prototype 4 and Prototype 5.1 bundles"
    )


def analyze_run_dir(
    run_dir: Path,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text())
    kind = detect_kind(run_dir)
    if kind == "prototype51":
        payload = json.loads(
            (run_dir / "diagnostics" / "preference_scores.json").read_text()
        )
        statistics = prototype51_statistics(
            payload["rows"],
            primary_scoring_mode=str(
                config.get("scoring", {}).get("primary", "option_text_logprob_margin")
            ),
            kl_max=float(config.get("gates", {}).get("kl_max_for_effect_summary", 0.25)),
            n_resamples=n_resamples,
            n_permutations=n_permutations,
            seed=seed,
        )
        headline = statistics["expected_effect"]["bootstrap_by_pair"]["estimate"]
    else:
        payload = json.loads(
            (run_dir / "diagnostics" / "matching_token_scores.json").read_text()
        )
        statistics = prototype4_statistics(
            payload["rows"],
            n_resamples=n_resamples,
            n_permutations=n_permutations,
            seed=seed,
        )
        headline = statistics["target_delta"]["bootstrap"]["estimate"]
    return {
        "run_dir": str(run_dir),
        "kind": kind,
        "experiment": config.get("experiment"),
        "run_seed": config.get("seed"),
        "headline_estimate": headline,
        "statistics": statistics,
    }


def analyze(
    run_dirs: list[Path],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> dict[str, Any]:
    reports = [
        analyze_run_dir(
            run_dir, n_resamples=n_resamples, n_permutations=n_permutations, seed=seed
        )
        for run_dir in run_dirs
    ]
    kinds = {report["kind"] for report in reports}
    combined: dict[str, Any] = {
        "disclaimer": STATISTICS_DISCLAIMER,
        "runs": reports,
    }
    if len(reports) > 1:
        if len(kinds) != 1:
            raise ValueError(
                "Multi-run analysis requires bundles of the same prototype; got: "
                + ", ".join(sorted(kinds))
            )
        combined["multi_seed"] = multi_seed_summary(
            {report["run_seed"]: report["headline_estimate"] for report in reports}
        )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute bootstrap CIs and permutation nulls for Prototype 4 / 5.1 "
            "run bundles (additional reporting; does not touch gates)"
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Run bundle directory; repeat for a multi-seed summary",
    )
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Also write stats_summary.json into each analyzed run directory",
    )
    args = parser.parse_args()
    report = analyze(
        args.run_dir,
        n_resamples=args.n_resamples,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    if args.write:
        for run_report in report["runs"]:
            path = Path(run_report["run_dir"]) / "stats_summary.json"
            path.write_text(json.dumps(run_report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

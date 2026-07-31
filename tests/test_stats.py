import math

import numpy as np
import pytest

from functional_emotions.stats import (
    bootstrap_mean_ci,
    multi_seed_summary,
    paired_difference_stats,
    sign_flip_pvalue,
)


def test_bootstrap_mean_ci_recovers_point_estimate_and_brackets_mean():
    rng = np.random.default_rng(7)
    values = rng.normal(loc=0.5, scale=0.1, size=200).tolist()

    result = bootstrap_mean_ci(values, seed=3)

    assert result["estimate"] == pytest.approx(float(np.mean(values)))
    assert result["ci_low"] < result["estimate"] < result["ci_high"]
    assert result["ci_low"] > 0.4
    assert result["ci_high"] < 0.6
    assert result["n_clusters"] == 200


def test_bootstrap_mean_ci_is_deterministic_given_seed():
    values = [0.1, 0.4, -0.2, 0.9, 0.3]

    first = bootstrap_mean_ci(values, seed=11)
    second = bootstrap_mean_ci(values, seed=11)

    assert first == second


def test_bootstrap_mean_ci_drops_nonfinite_values():
    result = bootstrap_mean_ci([1.0, math.nan, 3.0, math.inf], seed=0)

    assert result["estimate"] == pytest.approx(2.0)
    assert result["n_values"] == 2
    assert result["n_dropped_nonfinite"] == 2


def test_cluster_bootstrap_is_wider_than_row_bootstrap_for_correlated_clusters():
    # Two tight clusters far apart: row-level resampling underestimates the
    # uncertainty of the cluster-level mean.
    values = [1.0, 1.01, 0.99, -1.0, -1.01, -0.99]
    clusters = ["a", "a", "a", "b", "b", "b"]

    plain = bootstrap_mean_ci(values, seed=5)
    clustered = bootstrap_mean_ci(values, cluster_ids=clusters, seed=5)

    assert clustered["n_clusters"] == 2
    plain_width = plain["ci_high"] - plain["ci_low"]
    clustered_width = clustered["ci_high"] - clustered["ci_low"]
    assert clustered_width > plain_width


def test_bootstrap_with_single_cluster_reports_nan_interval():
    result = bootstrap_mean_ci([1.0, 2.0], cluster_ids=["only", "only"], seed=0)

    assert result["estimate"] == pytest.approx(1.5)
    assert math.isnan(result["ci_low"]) and math.isnan(result["ci_high"])


def test_sign_flip_exact_enumeration_for_small_cluster_counts():
    # Three positive clusters: only the all-positive and all-negative flips
    # reach |observed|, so the exact two-sided p-value is 2/8.
    values = [0.5, 0.5, 0.7, 0.7, 0.6, 0.6]
    clusters = ["a", "a", "b", "b", "c", "c"]

    result = sign_flip_pvalue(values, cluster_ids=clusters)

    assert result["method"] == "exact"
    assert result["n_permutations"] == 8
    assert result["p_value"] == pytest.approx(2 / 8)


def test_sign_flip_sampled_for_many_clusters_detects_strong_effect():
    rng = np.random.default_rng(0)
    values = (rng.normal(loc=1.0, scale=0.2, size=64)).tolist()

    result = sign_flip_pvalue(values, n_permutations=999, seed=1)

    assert result["method"] == "sampled"
    assert result["p_value"] < 0.01


def test_sign_flip_null_effect_has_large_p_value():
    values = [0.3, -0.31, 0.12, -0.11, 0.05, -0.06, 0.2, -0.19]

    result = sign_flip_pvalue(values, seed=2)

    assert result["p_value"] > 0.5


def test_paired_difference_stats_reports_mean_and_p_value():
    values_a = [1.0, 1.2, 0.8, 1.1]
    values_b = [0.5, 0.7, 0.4, 0.6]

    result = paired_difference_stats(values_a, values_b, seed=4)

    assert result["mean_difference"]["estimate"] == pytest.approx(0.475)
    assert result["sign_flip"]["observed_mean"] == pytest.approx(0.475)


def test_paired_difference_requires_equal_lengths():
    with pytest.raises(ValueError):
        paired_difference_stats([1.0], [1.0, 2.0])


def test_multi_seed_summary_for_three_seeds():
    summary = multi_seed_summary({42: 0.3, 43: 0.4, 44: 0.35})

    assert summary["n_seeds"] == 3
    assert summary["mean"] == pytest.approx(0.35)
    assert summary["all_same_sign"] is True
    assert summary["t_ci_low"] < 0.35 < summary["t_ci_high"]


def test_multi_seed_summary_ignores_nonfinite_estimates():
    summary = multi_seed_summary({1: 0.5, 2: math.nan})

    assert summary["n_seeds"] == 1
    assert summary["mean"] == pytest.approx(0.5)
    assert math.isnan(summary["std"])

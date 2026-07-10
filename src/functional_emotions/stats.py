"""Shared statistics helpers: cluster bootstrap CIs, permutation nulls, multi-seed.

This module is an additional reporting layer. It is never consulted by any
prototype's pre-registered hard gates, and adding it to a runner's metrics must
not change gate outcomes. All routines are deterministic given a seed and run
on plain Python/numpy without model weights.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


DEFAULT_RESAMPLES = 2000
DEFAULT_PERMUTATIONS = 10000
DEFAULT_CONFIDENCE = 0.95

# Two-sided 95% Student-t critical values for small degrees of freedom, used by
# the multi-seed summary (3-5 seeds means 2-4 degrees of freedom).
_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def _finite_values_and_clusters(
    values: Sequence[float], cluster_ids: Sequence[Any] | None
) -> tuple[np.ndarray, list[Any]]:
    array = np.asarray(list(values), dtype=float)
    if cluster_ids is None:
        ids: list[Any] = list(range(len(array)))
    else:
        ids = list(cluster_ids)
    if len(ids) != len(array):
        raise ValueError("cluster_ids must match values in length")
    mask = np.isfinite(array)
    return array[mask], [ids[i] for i in range(len(ids)) if mask[i]]


def _cluster_sums_counts(
    values: np.ndarray, cluster_ids: list[Any]
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    order: dict[Any, int] = {}
    for cid in cluster_ids:
        order.setdefault(cid, len(order))
    sums = np.zeros(len(order))
    counts = np.zeros(len(order))
    for value, cid in zip(values, cluster_ids, strict=True):
        index = order[cid]
        sums[index] += value
        counts[index] += 1
    return sums, counts, list(order)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    cluster_ids: Sequence[Any] | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile bootstrap CI for the mean, resampling whole clusters.

    ``cluster_ids`` groups rows that share a sampling unit (a story, a base
    activity pair, a prompt). Resampling clusters with replacement keeps
    within-cluster correlation intact; with ``cluster_ids=None`` every row is
    its own cluster and this reduces to an ordinary bootstrap. Non-finite
    values are dropped and counted.
    """

    raw_count = len(list(values))
    finite, ids = _finite_values_and_clusters(values, cluster_ids)
    result: dict[str, Any] = {
        "n_values": int(len(finite)),
        "n_dropped_nonfinite": int(raw_count - len(finite)),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
    }
    if len(finite) == 0:
        result.update(
            {"estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n_clusters": 0}
        )
        return result
    sums, counts, _ = _cluster_sums_counts(finite, ids)
    n_clusters = len(sums)
    result["n_clusters"] = int(n_clusters)
    result["estimate"] = float(finite.mean())
    if n_clusters < 2:
        result.update({"ci_low": math.nan, "ci_high": math.nan})
        return result
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))
    resampled = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    result["ci_low"] = float(np.quantile(resampled, alpha))
    result["ci_high"] = float(np.quantile(resampled, 1.0 - alpha))
    return result


def sign_flip_pvalue(
    values: Sequence[float],
    *,
    cluster_ids: Sequence[Any] | None = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """Two-sided cluster sign-flip permutation test of mean == 0.

    The null hypothesis is that effects are symmetric around zero, so flipping
    the sign of every value in a cluster is distribution-preserving. When the
    number of clusters is small enough that all sign assignments fit within
    ``n_permutations``, the test enumerates them exactly and reports
    ``method: "exact"``; otherwise it samples flips and reports
    ``method: "sampled"`` (with the +1 correction so p is never zero).
    """

    finite, ids = _finite_values_and_clusters(values, cluster_ids)
    if len(finite) == 0:
        return {"p_value": math.nan, "n_values": 0, "n_clusters": 0, "method": "none"}
    sums, counts, _ = _cluster_sums_counts(finite, ids)
    n_clusters = len(sums)
    total = counts.sum()
    observed = abs(sums.sum() / total)
    if n_clusters <= 30 and 2**n_clusters <= n_permutations:
        signs = np.array(
            [[1 if (mask >> bit) & 1 else -1 for bit in range(n_clusters)]
             for mask in range(2**n_clusters)],
            dtype=float,
        )
        flipped = np.abs(signs @ sums) / total
        p_value = float(np.mean(flipped >= observed - 1e-12))
        method = "exact"
        used = int(2**n_clusters)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, n_clusters))
        flipped = np.abs(signs @ sums) / total
        exceed = int(np.sum(flipped >= observed - 1e-12))
        p_value = float((exceed + 1) / (n_permutations + 1))
        method = "sampled"
        used = int(n_permutations)
    return {
        "p_value": p_value,
        "observed_mean": float(sums.sum() / total),
        "n_values": int(len(finite)),
        "n_clusters": int(n_clusters),
        "n_permutations": used,
        "method": method,
    }


def paired_difference_stats(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    cluster_ids: Sequence[Any] | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> dict[str, Any]:
    """CI and sign-flip p-value for the mean of paired differences a - b."""

    array_a = np.asarray(list(values_a), dtype=float)
    array_b = np.asarray(list(values_b), dtype=float)
    if array_a.shape != array_b.shape:
        raise ValueError("Paired difference requires equal-length inputs")
    differences = array_a - array_b
    return {
        "mean_difference": bootstrap_mean_ci(
            differences,
            cluster_ids=cluster_ids,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        ),
        "sign_flip": sign_flip_pvalue(
            differences,
            cluster_ids=cluster_ids,
            n_permutations=n_permutations,
            seed=seed,
        ),
    }


def multi_seed_summary(estimates_by_seed: dict[Any, float]) -> dict[str, Any]:
    """Combine one scalar estimate per seed (intended for 3-5 seeds).

    Reports the per-seed values, their mean, sample standard deviation, range,
    sign agreement, and a Student-t 95% interval on the across-seed mean when
    at least two finite seeds are available.
    """

    finite = {
        str(seed): float(value)
        for seed, value in estimates_by_seed.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    values = np.array(list(finite.values()), dtype=float)
    summary: dict[str, Any] = {
        "per_seed": finite,
        "n_seeds": int(len(values)),
    }
    if len(values) == 0:
        summary.update({"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan})
        return summary
    summary.update(
        {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "all_same_sign": bool(np.all(values > 0) or np.all(values < 0)),
        }
    )
    if len(values) >= 2:
        std = float(values.std(ddof=1))
        df = len(values) - 1
        critical = _T_CRITICAL_95.get(df, 1.96)
        half_width = critical * std / math.sqrt(len(values))
        summary.update(
            {
                "std": std,
                "t_ci_low": float(values.mean() - half_width),
                "t_ci_high": float(values.mean() + half_width),
                "t_ci_confidence": 0.95,
            }
        )
    else:
        summary["std"] = math.nan
    return summary

import pytest

pytest.importorskip("torch")

from functional_emotions.prototype0 import spearman


def test_spearman_detects_monotonic_effect():
    assert spearman([0.5, 1.0, 2.0], [1.0, 2.0, 4.0]) == pytest.approx(1.0)

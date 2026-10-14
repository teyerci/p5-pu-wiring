"""Tests for the H3 ranking-agreement measures."""
import numpy as np
import pytest

from src.ranking import ranking_agreement, top_k, top_k_overlap


def test_top_k_and_overlap_by_hand():
    a = np.array([0.9, 0.1, 0.5, 0.7, 0.3])
    b = np.array([0.2, 0.8, 0.6, 0.7, 0.1])
    assert list(top_k(a, 2)) == [0, 3] and list(top_k(b, 2)) == [1, 3]
    assert top_k_overlap(a, b, 2) == 0.5 and top_k_overlap(a, a, 3) == 1.0
    with pytest.raises(ValueError):
        top_k(a, 0)


def test_monotone_transform_agrees_fully_and_reversal_does_not():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    r = ranking_agreement(x, 0.6 / (1 + np.exp(-2 * x)), coef_a=[1.0, -2.0, 0.5], coef_b=[0.5, -1.0, 0.25], ks=(10, 100))
    assert r["spearman"] == pytest.approx(1.0) and r["top10_overlap"] == 1.0 and r["top100_overlap"] == 1.0
    assert r["coef_spearman"] == pytest.approx(1.0) and r["coef_abs_spearman"] == pytest.approx(1.0)
    rev = ranking_agreement(x, -x, ks=(10,))
    assert rev["spearman"] == pytest.approx(-1.0) and rev["top10_overlap"] == 0.0 and "coef_spearman" not in rev
    with pytest.raises(ValueError):
        ranking_agreement(x, x[:-1])

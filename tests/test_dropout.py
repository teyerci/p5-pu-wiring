"""Tests for the dropout thinning and the realised label frequency."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from src.dropout import realised_c, thin


def pairs(counts):
    idx = pd.MultiIndex.from_tuples([(i, i + 1) for i in range(len(counts))], names=["pre_id", "post_id"])
    return pd.Series(counts, index=idx, name="syn")


def test_thin_is_deterministic_never_adds_and_q1_is_identity():
    p = pairs([10, 25, 3, 60])
    a, b = thin(p, 0.7, seed=1), thin(p, 0.7, seed=1)
    pd.testing.assert_series_equal(a, b)
    assert (a <= p.reindex(a.index)).all()
    pd.testing.assert_series_equal(thin(p, 1.0), p)
    c = thin(p, 0.5, mode="connection", seed=2)
    assert set(c.to_numpy()) <= set(p.to_numpy())             # whole connections kept or dropped
    with pytest.raises(ValueError):
        thin(p, 0.0)


def test_realised_c_matches_the_binomial_tail_and_is_flat_for_connection_dropout():
    p = pairs([10] * 4000 + [30] * 4000)
    t = realised_c(p, 0.8, tau=10, mode="synapse", seeds=(0, 1))
    assert t.loc["10–19", "exact"] == pytest.approx(binom.sf(9, 10, 0.8))   # 0.8 ** 10
    assert abs(t.loc["10–19", "sim_mean"] - 0.8 ** 10) < 0.02
    assert t.loc["20–49", "exact"] > 0.999
    f = realised_c(p, 0.8, tau=10, mode="connection", seeds=(0, 1))
    assert np.allclose(f["exact"].dropna(), 0.8) and abs(f.loc["all positives", "sim_mean"] - 0.8) < 0.02

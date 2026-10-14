"""Tests for the dropout-benchmark helpers."""
import pandas as pd
import pytest

from src.benchmark import edges_from_rows, score, thin_rows


def rows():
    return pd.DataFrame({"pre_id": [1, 1, 2, 3], "post_id": [2, 2, 3, 1],
                         "neuropil": ["FB", "EB", "FB", "NO"], "syn_count": [30, 20, 5, 12]})


def test_synapse_thinning_never_adds_and_connection_dropout_keeps_whole_pairs():
    er = rows()
    t = thin_rows(er, 0.5, "synapse", seed=1).merge(er, on=["pre_id", "post_id", "neuropil"], suffixes=("", "_orig"))
    assert (t.syn_count <= t.syn_count_orig).all()
    for seed in range(20):
        c = thin_rows(er, 0.5, "connection", seed=seed)
        kept = set(zip(c.pre_id, c.post_id))
        assert ((1, 2) in kept) == (len(c[(c.pre_id == 1) & (c.post_id == 2)]) == 2)   # both neuropil rows or none
        assert (c.syn_count == er.set_index(["pre_id", "post_id", "neuropil"]).loc[
            list(zip(c.pre_id, c.post_id, c.neuropil)), "syn_count"].to_numpy()).all()
    with pytest.raises(ValueError):
        thin_rows(er, 0.5, "other")


def test_edges_from_rows_and_weighted_scores():
    e = edges_from_rows(rows()).set_index(["pre_id", "post_id"])["syn_count"]
    assert e.to_dict() == {(1, 2): 50, (2, 3): 5, (3, 1): 12}
    sc = score(pd.Series([1.0, 0.0, 0.0]).to_numpy(), pd.Series([0.9, 0.1, 0.2]).to_numpy(),
               pd.Series([1.0, 2.0, 2.0]).to_numpy())
    assert sc["auc_y"] == 1.0 and abs(sc["mean_y"] - 0.2) < 1e-12

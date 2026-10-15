"""Tests for the cache helper and the validators, on hand-built toy tables."""
import pandas as pd
import pytest

import config
from src.cache import cached
from src.fetch_fafb import _cast, recode
from src.validate import (assert_interchangeable, check_plausibility, check_references,
                          validate_edges, validate_neurons)


def toy_neurons(**overrides):
    df = pd.DataFrame({
        "neuron_id": [1, 2, 3], "dataset": "fafb", "release": "783",
        "cell_type": ["A", "A", None], "cell_class": None, "cell_sub_class": None,
        "super_class": ["central", "central", "unknown"], "side": ["left", "right", "unknown"],
        "region": ["FB", "FB", "unknown"], "nt_type": ["gaba", "acetylcholine", "unknown"],
        "nt_confidence": [0.9, 0.8, None], "n_pre": pd.NA, "n_post": [10, 10, pd.NA],
        "flywire_type": pd.NA, "manc_type": pd.NA, "hemibrain_type": None, "dimorphism": None,
        "status": None, "hemilineage": None, "nt_known": None,
    })
    for k, v in overrides.items():
        df[k] = v
    return _cast(df, config.NEURON_SCHEMA)


def toy_edges(**overrides):
    df = pd.DataFrame({"pre_id": [1, 2], "post_id": [2, 1], "syn_count": [5, 1],
                       "dataset": "fafb", "release": "783", "min_syn_fetched": 1})
    for k, v in overrides.items():
        df[k] = v
    return _cast(df, config.EDGE_SCHEMA)


def test_valid_tables_pass():
    n, e = toy_neurons(), toy_edges()
    validate_neurons(n)
    validate_edges(e)
    check_references(n, e)
    check_plausibility(n, e)


def test_float_id_is_caught():
    n = toy_neurons()
    n["neuron_id"] = n["neuron_id"].astype("float64")
    with pytest.raises(ValueError, match="neuron_id"):
        validate_neurons(n)


def test_duplicate_key_is_caught():
    n = toy_neurons(neuron_id=[1, 1, 3])
    with pytest.raises(ValueError, match="repeated"):
        validate_neurons(n)


def test_null_in_non_nullable_column_is_caught():
    n = toy_neurons()
    n["region"] = pd.array(["FB", None, "FB"], dtype="string")
    with pytest.raises(ValueError, match="region"):
        validate_neurons(n)


def test_unexpected_column_is_caught():
    n = toy_neurons()
    n["extra"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        validate_neurons(n)


def test_dangling_edge_id_is_caught():
    with pytest.raises(ValueError, match="post_id"):
        check_references(toy_neurons(), toy_edges(post_id=[2, 99]))


def test_below_fetch_threshold_is_caught():
    with pytest.raises(ValueError, match="syn_count"):
        validate_edges(toy_edges(min_syn_fetched=2))


def test_plausibility_catches_inputs_above_n_post():
    with pytest.raises(ValueError, match="n_post"):
        check_plausibility(toy_neurons(), toy_edges(syn_count=[50, 1]))


def test_recode_raises_on_unmapped_value():
    with pytest.raises(ValueError, match="side"):
        recode(pd.Series(["left", "L"]), {"left": "left"}, "side")
    assert list(recode(pd.Series(["left", None]), {"left": "left"}, "side")) == ["left", "unknown"]


def test_interchangeable():
    assert_interchangeable(toy_neurons(), toy_neurons(dataset="malecns", release="v1.0"))


def test_cache_fetches_once_and_refreshes_on_request(tmp_path):
    calls = []

    def fetch():
        calls.append(1)
        return toy_edges()

    path = tmp_path / "fafb_edges_783_buhmann_min1.parquet"
    first = cached(path, fetch, validate_edges)
    second = cached(path, fetch, validate_edges)
    assert len(calls) == 1
    assert_interchangeable(first, second)          # dtypes survive the parquet round trip
    cached(path, fetch, validate_edges, force_refresh=True)
    assert len(calls) == 2


def test_recode_none_mapping_means_unknown():
    assert list(recode(pd.Series(["vnc_intrinsic", "cb_intrinsic"]),
                       {"vnc_intrinsic": None, "cb_intrinsic": "central"}, "super_class")) == ["unknown", "central"]


def test_malecns_region_lets_unspecified_win_and_breaks_ties_alphabetically(monkeypatch):
    import numpy as np
    import pyarrow as pa
    from src import fetch_malecns

    # body 1: 3 synapses in CentralBrain-unspecified, 2 in FB (as pre and post) -> CentralBrain-unspecified
    # body 2: one synapse each in PB and EB (a tie) -> EB
    # body 3: <unspecified> twice and one unlabelled, against 1 in FB -> UNASGD (no ROI is FlyWire's UNASGD)
    rows = [(1, 9, "CentralBrain-unspecified")] * 3 + [(1, 9, "FB"), (9, 1, "FB"),
            (2, 9, "PB"), (9, 2, "EB"), (3, 9, "<unspecified>"), (9, 3, "<unspecified>"), (3, 9, None),
            (3, 9, "FB")]
    pre, post, roi = zip(*rows)
    table = pa.table({"body_pre": pa.array(pre, pa.int64()), "body_post": pa.array(post, pa.int64()),
                      "primary_post": pa.array(roi, pa.string()).dictionary_encode()})
    monkeypatch.setattr(fetch_malecns, "_partners", lambda columns: table.select(columns))
    region = fetch_malecns._region_from_all_synapses(np.array([1, 2, 3]))
    assert region.to_dict() == {1: "CentralBrain-unspecified", 2: "EB", 3: "UNASGD"}


def test_fafb_region_spells_none_as_unasgd_and_lets_it_win(tmp_path, monkeypatch):
    import numpy as np
    from src import fetch_fafb

    # neuron 1: 3 None, 2 FB -> UNASGD; neuron 2: 2 None, 2 GNG, a tie -> GNG (alphabetical); neuron 3 is not asked for
    pre = pd.DataFrame({"pre_pt_root_id": [1, 1, 2, 3], "neuropil": ["None", "FB", "GNG", "None"], "count": [2, 2, 2, 9]})
    post = pd.DataFrame({"post_pt_root_id": [1, 2], "neuropil": ["None", "None"], "count": [1, 1]})
    pre.to_feather(tmp_path / "pre.feather")
    post.to_feather(tmp_path / "post.feather")
    monkeypatch.setattr(fetch_fafb, "raw_file", lambda name: tmp_path / {"neuropil_pre": "pre.feather",
                                                                        "neuropil_post": "post.feather"}[name])
    region = fetch_fafb._region_from_all_synapses(np.array([1, 2]))
    assert region.to_dict() == {1: "UNASGD", 2: "GNG"}


def _malecns_raw(tmp_path, monkeypatch):
    """Tiny stand-ins for the MaleCNS bulk files, wired into src.fetch_malecns."""
    import pyarrow as pa
    from src import fetch_malecns
    ann = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5], "status": ["Traced"] * 4 + ["Orphan"],
        "somaSide": ["L", None, None, "M", "R"], "rootSide": ["R", "R", "unknown", None, "R"],
        "type": ["A", "B", None, "D", "E"], "class": None, "subclass": None,
        "superclass": ["cb_intrinsic", "vnc_intrinsic", None, "cb_efferent", "cb_intrinsic"],
        "flywireType": ["A", None, None, None, None], "mancType": None, "hemibrainType": None,
        "dimorphism": None, "itoleeHl": None})
    nt = pd.DataFrame({"body": [1, 2, 3], "predicted_nt": ["gaba", "unclear", "acetylcholine"],
                       "predicted_nt_confidence": [0.9, 0.4, 0.8], "ground_truth": ["gaba", None, None]})
    stats = pd.DataFrame({"body": [1, 2, 3, 4], "pre": [5, 1, 0, 2], "post": [7, 3, 1, 0]})
    paths = {"annotations": ann, "neurotransmitters": nt, "body_stats": stats}
    for k, df in paths.items():
        df.to_feather(tmp_path / f"{k}.feather")
    monkeypatch.setattr(fetch_malecns, "raw_file", lambda key: tmp_path / f"{key}.feather")
    syn = pa.table({"body_pre": pa.array([1, 1, 2, 2, 5], pa.int64()), "body_post": pa.array([2, 2, 1, 1, 1], pa.int64()),
                    "primary_post": pa.array(["FB", None, "EB", "EB", "FB"], pa.string())})
    monkeypatch.setattr(fetch_malecns, "_partners", lambda columns: syn.select(columns))
    return fetch_malecns


def test_malecns_neurons_side_nt_and_super_class(tmp_path, monkeypatch):
    fm = _malecns_raw(tmp_path, monkeypatch)
    n = fm.fetch_neurons().set_index("neuron_id")
    assert list(n.index) == [1, 2, 3, 4]                                    # Traced only
    assert list(n["side"]) == ["left", "right", "unknown", "center"]       # somaSide first, then rootSide
    assert list(n["nt_type"]) == ["gaba", "unknown", "acetylcholine", "unknown"]
    assert n.loc[2, "nt_confidence"] != n.loc[2, "nt_confidence"]          # "unclear": confidence masked (NaN)
    assert n.loc[1, "nt_known"] == "gaba" and pd.isna(n.loc[3, "nt_known"])
    assert list(n["super_class"]) == ["central", "unknown", "unknown", "motor"]
    # neuron 1: EB 2 (as post) ties FB 2 (1 as pre, 1 from untraced body 5: all partners count);
    # the null label is ignored and the tie goes to the alphabetically first neuropil
    assert n.loc[1, "region"] == "EB"


def test_malecns_edgeroi_keeps_missing_labels_as_unassigned(tmp_path, monkeypatch):
    fm = _malecns_raw(tmp_path, monkeypatch)
    monkeypatch.setattr(fm, "traced_ids", lambda: __import__("numpy").array([1, 2, 3, 4]))
    er = fm.fetch_edgeroi()
    got = {(r.pre_id, r.post_id, r.neuropil): r.syn_count for r in er.itertuples()}
    assert got == {(1, 2, "FB"): 1, (1, 2, "UNASGD"): 1, (2, 1, "EB"): 2}  # body 5 is not Traced

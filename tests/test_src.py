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
        "nt_type_corrected": ["gaba", "acetylcholine", "unknown"], "nt_correction_source": "predicted",
        "super_class_source": None,
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


def test_cached_rebuilds_a_file_from_an_older_schema(tmp_path):
    calls = []

    def fetch():
        calls.append(1)
        return toy_edges()

    path = tmp_path / "edges.parquet"
    toy_edges().drop(columns="dataset").to_parquet(path, index=False)
    assert "dataset" in cached(path, fetch, validate_edges, columns=["dataset"]).columns
    assert len(calls) == 1
    cached(path, fetch, validate_edges, columns=["dataset"])
    assert len(calls) == 1


def test_recode_none_mapping_means_unknown():
    assert list(recode(pd.Series(["vnc_intrinsic", "cb_intrinsic"]),
                       {"vnc_intrinsic": None, "cb_intrinsic": "central"}, "super_class")) == ["unknown", "central"]


def test_fafb_neurons_apply_corrections_in_order(tmp_path, monkeypatch):
    import numpy as np
    from src import fetch_fafb

    # 9 is outside the release and dropped; 3 has no annotation row and gets unknown labels and a real region
    ann = pd.DataFrame({"root_id": [1, 2, 9], "cell_type": ["KCg", "A", "B"], "cell_class": None, "cell_sub_class": None,
                        "super_class": ["central", "sensory_ascending", "optic"], "side": ["left", "right", "na"],
                        "top_nt": ["dopamine", "gaba", "gaba"], "top_nt_conf": [0.9, 0.8, 0.7], "hemibrain_type": None,
                        "dimorphism": None, "status": None, "ito_lee_hemilineage": None,
                        "known_nt": ["acetylcholine", None, None]})
    ann.to_csv(tmp_path / "annotations", sep="\t", index=False)
    np.save(tmp_path / "proofread_ids.npy", np.array([1, 2, 3]))
    # 1: None outweighs FB but is never a region; 2: PB and EB tie -> EB; 3: GNG; 9 is a fragment outside the set
    pd.DataFrame({"pre_pt_root_id": [1, 1, 2, 3, 9], "neuropil": ["None", "FB", "PB", "GNG", "FB"],
                  "count": [5, 1, 2, 1, 9]}).to_feather(tmp_path / "neuropil_pre")
    pd.DataFrame({"post_pt_root_id": [2], "neuropil": ["EB"], "count": [2]}).to_feather(tmp_path / "neuropil_post")
    monkeypatch.setattr(fetch_fafb, "raw_file", lambda key: tmp_path / {"proofread_ids": "proofread_ids.npy"}.get(key, key))
    n = fetch_fafb.fetch_neurons().set_index("neuron_id")
    assert sorted(n.index) == [1, 2, 3]
    assert n["region"].to_dict() == {1: "FB", 2: "EB", 3: "GNG"}
    assert n.loc[1, "nt_type"] == "dopamine" and n.loc[1, "nt_type_corrected"] == "acetylcholine"
    assert n.loc[1, "nt_correction_source"] == "known_nt" and n.loc[2, "nt_correction_source"] == "predicted"
    assert n.loc[2, "super_class"] == "sensory_ascending" and n.loc[2, "super_class_source"] == "sensory_ascending"
    assert n.loc[3, "super_class"] == "unknown" and n.loc[3, "nt_type_corrected"] == "unknown"
    validate_neurons(n.reset_index()[list(config.NEURON_SCHEMA)])


def _malecns_raw(tmp_path, monkeypatch):
    """Tiny stand-ins for the MaleCNS bulk files, wired into src.fetch_malecns."""
    import pyarrow as pa
    from src import fetch_malecns
    ann = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5], "status": ["Traced"] * 4 + ["Orphan"],
        "somaSide": ["L", None, None, "M", "R"], "rootSide": ["R", "R", "unknown", None, "R"],
        "type": ["A", "KCg", None, "D", "E"], "class": None, "subclass": None,
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
    # 1: EB and FB tie -> EB, <unspecified> never wins; 2: a brain neuropil beats a larger nerve-cord one;
    # 3: nerve cord only -> unknown; 4: no row -> unknown
    roi = pd.DataFrame({"bodyId": [1, 1, 1, 2, 2, 3], "roi": ["FB", "EB", "<unspecified>", "PB", "ANm", "ANm"],
                        "upstream": [1, 2, 9, 1, 5, 1], "downstream": [1, 0, 0, 0, 5, 1]})
    roi_path = tmp_path / "roi.parquet"
    roi.to_parquet(roi_path)
    monkeypatch.setattr(fetch_malecns, "MALECNS_ROI_COUNTS", roi_path)
    return fetch_malecns


def test_malecns_neurons_side_nt_and_super_class(tmp_path, monkeypatch):
    fm = _malecns_raw(tmp_path, monkeypatch)
    n = fm.fetch_neurons().set_index("neuron_id")
    assert list(n.index) == [1, 2, 3, 4]                                    # Traced only
    assert list(n["side"]) == ["left", "right", "unknown", "center"]       # somaSide first, then rootSide
    assert list(n["nt_type"]) == ["gaba", "unknown", "acetylcholine", "unknown"]
    assert n.loc[2, "nt_confidence"] != n.loc[2, "nt_confidence"]          # "unclear": confidence masked (NaN)
    assert n.loc[1, "nt_known"] == "gaba" and pd.isna(n.loc[3, "nt_known"])
    assert list(n["super_class"]) == ["central", "vnc_intrinsic", "unknown", "efferent"]
    assert n.loc[4, "super_class_source"] == "cb_efferent"
    assert list(n["region"]) == ["EB", "PB", "unknown", "unknown"]
    assert n.loc[2, "nt_type"] == "unknown" and n.loc[2, "nt_type_corrected"] == "acetylcholine"   # KCg
    assert list(n["nt_correction_source"]) == ["predicted", "kc_override", "predicted", "predicted"]


def test_malecns_brain_part_caches_and_crops(tmp_path, monkeypatch):
    fm = _malecns_raw(tmp_path, monkeypatch)
    monkeypatch.setattr(fm, "traced_ids", lambda: __import__("numpy").array([1, 2, 3, 4]))
    refreshed = []
    monkeypatch.setattr(fm, "load_neurons", lambda force=False: refreshed.append(force) or fm.fetch_neurons())
    monkeypatch.setattr(fm, "load_edgeroi", lambda: fm.fetch_edgeroi())
    monkeypatch.setattr(fm, "cache_file", lambda ds, table, *a, **k: tmp_path / f"{table}.parquet")
    n, e, er, report = fm.load_brain()
    assert list(n["neuron_id"]) == [1, 2]                                  # 3 and 4 have no brain connection
    # the unassigned row of pair (1, 2) counts as neither brain nor nerve cord, so its weight falls to 1
    assert {(r.pre_id, r.post_id): r.syn_count for r in e.itertuples()} == {(1, 2): 1, (2, 1): 2}
    assert report["neurons_dropped"] == 2
    again = fm.load_brain()
    assert_interchangeable(n, again[0])
    assert again[3] == report
    fm.load_brain(force_refresh=True)
    assert refreshed == [False, True]                                      # a forced rebuild reaches the neurons


def test_malecns_edgeroi_keeps_missing_labels_as_unassigned(tmp_path, monkeypatch):
    fm = _malecns_raw(tmp_path, monkeypatch)
    monkeypatch.setattr(fm, "traced_ids", lambda: __import__("numpy").array([1, 2, 3, 4]))
    er = fm.fetch_edgeroi()
    got = {(r.pre_id, r.post_id, r.neuropil): r.syn_count for r in er.itertuples()}
    assert got == {(1, 2, "FB"): 1, (1, 2, "UNASGD"): 1, (2, 1, "EB"): 2}  # body 5 is not Traced

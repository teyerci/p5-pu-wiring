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

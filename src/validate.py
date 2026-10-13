"""The eight validators from the Week 2 Data Conventions.

Every check reads the schema from config.py, so the schema has one definition.
Each raises a ValueError that names the offending column.
"""
from __future__ import annotations

import pandas as pd

import config

_VOCAB = {
    "dataset": config.DATASET_VALUES,
    "side": config.SIDE_VALUES,
    "nt_type": config.NT_VALUES,
    "super_class": config.SUPER_CLASS_VALUES,
}


def _expected_dtype(spec):
    return spec if isinstance(spec, pd.CategoricalDtype) else pd.api.types.pandas_dtype(spec)


def check_columns(df: pd.DataFrame, schema: dict) -> None:
    missing = [c for c in schema if c not in df.columns]
    extra = [c for c in df.columns if c not in schema]
    if missing or extra:
        raise ValueError(f"columns: missing {missing}, unexpected {extra}")
    if list(df.columns) != list(schema):
        raise ValueError(f"columns: order differs from config ({list(df.columns)})")


def check_dtypes(df: pd.DataFrame, schema: dict) -> None:
    for col, (spec, _) in schema.items():
        want = _expected_dtype(spec)
        if df[col].dtype != want:
            raise ValueError(f"{col}: dtype {df[col].dtype}, expected {want}")


def check_nulls(df: pd.DataFrame, schema: dict) -> None:
    for col, (_, nullable) in schema.items():
        if not nullable and df[col].isna().any():
            raise ValueError(f"{col}: {int(df[col].isna().sum())} missing values in a non-nullable column")


def check_keys(df: pd.DataFrame, key: list[str]) -> None:
    dup = df.duplicated(subset=key)
    if dup.any():
        raise ValueError(f"{'+'.join(key)}: {int(dup.sum())} repeated keys")


def check_vocabulary(df: pd.DataFrame) -> None:
    for col, allowed in _VOCAB.items():
        if col in df:
            bad = set(df[col].dropna().astype(str)) - set(allowed)
            if bad:
                raise ValueError(f"{col}: values outside the vocabulary {sorted(bad)[:10]}")
    if config.REGION_VALUES is not None and "region" in df:
        bad = set(df["region"].dropna()) - set(config.REGION_VALUES)
        if bad:
            raise ValueError(f"region: values outside the vocabulary {sorted(bad)[:10]}")


def check_threshold(edges: pd.DataFrame) -> None:
    below = edges["syn_count"] < edges["min_syn_fetched"]
    if below.any():
        raise ValueError(f"syn_count: {int(below.sum())} rows below min_syn_fetched")


def check_references(neurons: pd.DataFrame, edges: pd.DataFrame) -> None:
    for ds, e in edges.groupby("dataset", observed=True):
        ids = pd.Index(neurons.loc[neurons["dataset"] == ds, "neuron_id"])
        for col in ("pre_id", "post_id"):
            missing = ~e[col].isin(ids)
            if missing.any():
                raise ValueError(f"{col}: {int(missing.sum())} ids with no neuron in dataset {ds}")


def check_plausibility(neurons: pd.DataFrame, edges: pd.DataFrame) -> None:
    """Input synapses in the edge table must not exceed n_post (skipped where n_post is missing)."""
    for ds, e in edges.groupby("dataset", observed=True):
        n = neurons[(neurons["dataset"] == ds) & neurons["n_post"].notna()].set_index("neuron_id")["n_post"]
        if n.empty:
            continue
        total_in = e.groupby("post_id")["syn_count"].sum()
        over = total_in.reindex(n.index).fillna(0) > n.astype("int64")
        if over.any():
            raise ValueError(f"n_post: {int(over.sum())} neurons with more input synapses than n_post ({ds})")


def validate_neurons(neurons: pd.DataFrame) -> None:
    check_columns(neurons, config.NEURON_SCHEMA)
    check_dtypes(neurons, config.NEURON_SCHEMA)
    check_nulls(neurons, config.NEURON_SCHEMA)
    check_keys(neurons, ["neuron_id", "dataset"])
    check_vocabulary(neurons)


def _validate_edge_table(edges: pd.DataFrame, schema: dict, key: list[str]) -> None:
    check_columns(edges, schema)
    check_dtypes(edges, schema)
    check_nulls(edges, schema)
    check_keys(edges, key)
    check_vocabulary(edges)
    check_threshold(edges)


def validate_edges(edges: pd.DataFrame) -> None:
    _validate_edge_table(edges, config.EDGE_SCHEMA, ["pre_id", "post_id", "dataset"])


def validate_edgeroi(edgeroi: pd.DataFrame) -> None:
    _validate_edge_table(edgeroi, config.EDGEROI_SCHEMA, ["pre_id", "post_id", "neuropil", "dataset"])


def validate_dataset(neurons: pd.DataFrame, edges: pd.DataFrame, edgeroi: pd.DataFrame | None = None) -> None:
    """All eight checks, including the two that need both tables."""
    validate_neurons(neurons)
    validate_edges(edges)
    check_references(neurons, edges)
    check_plausibility(neurons, edges)
    if edgeroi is not None:
        validate_edgeroi(edgeroi)
        check_references(neurons, edgeroi)


def assert_interchangeable(a: pd.DataFrame, b: pd.DataFrame) -> None:
    assert list(a.columns) == list(b.columns), "column sets differ"
    assert a.dtypes.equals(b.dtypes), "dtypes differ"

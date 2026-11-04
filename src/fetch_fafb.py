"""FAFB (FlyWire 783) fetch functions: neurons, edges, edgeroi.

Sources are the pinned bulk files (src/paths.py). Raw files are downloaded
once into data/raw/; the tables built from them go through the cache helper.
Nothing downstream sees FlyWire's own column names.
"""
from __future__ import annotations

import logging
import shutil
import urllib.request

import numpy as np
import pandas as pd

import config
from eda_shared.corrections import (add_nt_type_corrected, align_to_proofread, fafb_neuropil_counts,
                                    map_super_class, region_from_counts)
from src.cache import cached
from src.paths import FAFB_FILES, FAFB_RAW, cache_file
from src.validate import validate_edgeroi, validate_edges, validate_neurons

log = logging.getLogger(__name__)

DATASET = "fafb"
RELEASE = str(config.FAFB_MATERIALIZATION)
ANN_SOURCE = "ann" + config.FAFB_ANNOTATIONS_TAG.lstrip("v")      # ann3.2.0
SYN_SOURCE = config.FAFB_SYNAPSE_TABLE                              # buhmann

# Source value -> agreed vocabulary. An unmapped value raises (recode).
SIDE_MAP = {"left": "left", "right": "right", "center": "center", "na": "unknown"}
NT_MAP = {v: v for v in ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine"]}
def recode(series: pd.Series, mapping: dict, column: str) -> pd.Series:
    """Map source values to the vocabulary; an unmapped value raises, a None mapping means unknown."""
    unmapped = series[~series.isin(list(mapping)) & series.notna()].unique()
    if len(unmapped):
        raise ValueError(f"{column}: unmapped values {unmapped[:10]}")
    return series.map(mapping).fillna("unknown")


def raw_file(key: str):
    """Path to a raw FAFB file, downloading it on first use."""
    name, url = FAFB_FILES[key]
    path = FAFB_RAW / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", name)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url.format(name=name)) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(path)
    return path


def _connections() -> pd.DataFrame:
    return pd.read_feather(raw_file("connections"),
                           columns=["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"])


def _neuropil_counts(neuron_ids: np.ndarray) -> pd.DataFrame:
    """FlyWire's per-neuron neuropil counts for these neurons; the files also cover unproofread fragments."""
    pre = pd.read_feather(raw_file("neuropil_pre"))
    post = pd.read_feather(raw_file("neuropil_post"))
    return fafb_neuropil_counts(pre[pre["pre_pt_root_id"].isin(neuron_ids)],
                                post[post["post_pt_root_id"].isin(neuron_ids)])


def fetch_neurons() -> pd.DataFrame:
    """The 783 proofread neurons (Corrections 3), in the order eda_shared's README gives."""
    ann = pd.read_csv(raw_file("annotations"), sep="\t", low_memory=False,
                      dtype={"root_id": "int64"})
    sc = map_super_class(ann["super_class"], DATASET)                   # correction 2
    nt_type = recode(ann["top_nt"], NT_MAP, "nt_type")
    df = pd.DataFrame({
        "neuron_id": ann["root_id"].astype("int64"),
        "dataset": DATASET,
        "release": RELEASE,
        "cell_type": ann["cell_type"],
        "cell_class": ann["cell_class"],
        "cell_sub_class": ann["cell_sub_class"],
        "super_class": sc["super_class"].array,
        "side": recode(ann["side"], SIDE_MAP, "side"),
        "region": "unknown",
        "nt_type": nt_type,
        "nt_confidence": ann["top_nt_conf"].where(nt_type != "unknown"),
        "n_pre": pd.NA,       # not provided by FlyWire (Data Conventions: left missing)
        "n_post": pd.NA,
        "flywire_type": pd.NA,
        "manc_type": pd.NA,
        "hemibrain_type": ann["hemibrain_type"],
        "dimorphism": ann["dimorphism"],
        "status": ann["status"],
        "hemilineage": ann["ito_lee_hemilineage"],
        "nt_known": ann["known_nt"],
        "super_class_source": sc["super_class_source"].array,
    })
    df = df.astype({c: config.NEURON_SCHEMA[c][0] for c in df.columns})
    df, report = align_to_proofread(df, np.load(raw_file("proofread_ids")))      # correction 3
    log.info("neuron set: %s", report)
    ids = df["neuron_id"].to_numpy()
    reg = region_from_counts(ids, _neuropil_counts(ids)).set_index("neuron_id")  # correction 6
    df["region"] = df["neuron_id"].map(reg["region"]).astype("string")
    log.info("region: %d neurons tie between their top two neuropils", int(reg["tied"].sum()))
    return _cast(add_nt_type_corrected(df), config.NEURON_SCHEMA)                # correction 1


def _aggregate(conn: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    e = (conn.groupby(by, sort=False)["syn_count"].sum().reset_index()
         .rename(columns={"pre_pt_root_id": "pre_id", "post_pt_root_id": "post_id"}))
    self_conn = e["pre_id"] == e["post_id"]
    log.info("dropping %d self-connections", int(self_conn.sum()))
    e = e[~self_conn]
    e = e[e["syn_count"] >= config.FETCH_MIN_SYN]
    e["dataset"] = DATASET
    e["release"] = RELEASE
    e["min_syn_fetched"] = config.FETCH_MIN_SYN
    return e


def fetch_edges() -> pd.DataFrame:
    """One row per neuron pair; synapses summed over neuropils, then thresholded."""
    e = _aggregate(_connections(), ["pre_pt_root_id", "post_pt_root_id"])
    return _cast(e, config.EDGE_SCHEMA)


def fetch_edgeroi() -> pd.DataFrame:
    """One row per (pair, neuropil); syn_count is per neuropil."""
    e = _aggregate(_connections(), ["pre_pt_root_id", "post_pt_root_id", "neuropil"])
    return _cast(e, config.EDGEROI_SCHEMA)


def _cast(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, (dtype, _) in schema.items():
        out[col] = df[col].astype(dtype)
    return out.reset_index(drop=True)


def load_neurons(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "neurons", RELEASE, ANN_SOURCE),
                  fetch_neurons, validate_neurons, force_refresh, columns=config.NEURON_SCHEMA)


def load_edges(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "edges", RELEASE, SYN_SOURCE, min_syn=config.FETCH_MIN_SYN),
                  fetch_edges, validate_edges, force_refresh)


def load_edgeroi(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "edgeroi", RELEASE, SYN_SOURCE, min_syn=config.FETCH_MIN_SYN),
                  fetch_edgeroi, validate_edgeroi, force_refresh)

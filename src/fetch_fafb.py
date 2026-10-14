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
SUPER_CLASS_MAP = {v: v for v in ["sensory", "ascending", "descending", "motor", "endocrine",
                                  "visual_projection", "visual_centrifugal", "optic", "central"]}
# PROVISIONAL, to raise with the group: FAFB has a "sensory_ascending" super class
# (612 neurons) that config.SUPER_CLASS_VALUES does not list. Mapped to "sensory"
# so the sensory/afferent distinction survives; the original label is kept in
# cell_class where FlyWire gives one.
SUPER_CLASS_MAP["sensory_ascending"] = "sensory"


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


def _region_from_all_synapses(neuron_ids: np.ndarray) -> pd.Series:
    """Neuropil holding most of each neuron's synapses (inputs plus outputs).

    Uses FlyWire's per-neuron neuropil counts, which include synapses with
    non-proofread partners ("from all synapses"). Ties: the tie rule is a
    week 3 decision; until then the alphabetically first neuropil wins.
    """
    pre = pd.read_feather(raw_file("neuropil_pre")).rename(columns={"pre_pt_root_id": "neuron_id"})
    post = pd.read_feather(raw_file("neuropil_post")).rename(columns={"post_pt_root_id": "neuron_id"})
    both = pd.concat([pre, post])
    both = both[both["neuron_id"].isin(neuron_ids)]      # the files also cover unproofread fragments
    # FlyWire labels synapses outside every neuropil "None"; that is not a region,
    # so it never wins. A neuron with only such synapses gets region "unknown".
    unassigned = both["neuropil"].isin(["None", "UNASGD"])
    log.info("region: %d synapses outside any neuropil ignored", int(both.loc[unassigned, "count"].sum()))
    both = both[~unassigned]
    counts = (both.groupby(["neuron_id", "neuropil"])["count"].sum()
              .reset_index().sort_values(["neuron_id", "count", "neuropil"],
                                         ascending=[True, False, True]))
    top = counts.drop_duplicates("neuron_id")
    second = counts[counts.duplicated("neuron_id")].drop_duplicates("neuron_id").set_index("neuron_id")["count"]
    ties = int((top.set_index("neuron_id")["count"] == second.reindex(top["neuron_id"]).values).sum())
    log.info("region: %d neurons tie between their top two neuropils", ties)
    return top.set_index("neuron_id")["neuropil"]


def fetch_neurons() -> pd.DataFrame:
    ann = pd.read_csv(raw_file("annotations"), sep="\t", low_memory=False,
                      dtype={"root_id": "int64"})
    # Ids that appear in the connection table but have no annotation row are
    # added with unknown annotations (Data Conventions, first EDA check).
    conn = _connections()
    edge_ids = pd.unique(np.concatenate([conn["pre_pt_root_id"].to_numpy(), conn["post_pt_root_id"].to_numpy()]))
    extra = np.setdiff1d(edge_ids, ann["root_id"].to_numpy())
    if len(extra):
        log.info("adding %d edge ids without an annotation row", len(extra))
        ann = pd.concat([ann, pd.DataFrame({"root_id": extra})], ignore_index=True)

    nt_type = recode(ann["top_nt"], NT_MAP, "nt_type")
    df = pd.DataFrame({
        "neuron_id": ann["root_id"].astype("int64"),
        "dataset": DATASET,
        "release": RELEASE,
        "cell_type": ann["cell_type"],
        "cell_class": ann["cell_class"],
        "cell_sub_class": ann["cell_sub_class"],
        "super_class": recode(ann["super_class"], SUPER_CLASS_MAP, "super_class"),
        "side": recode(ann["side"], SIDE_MAP, "side"),
        "region": ann["root_id"].map(_region_from_all_synapses(ann["root_id"].to_numpy())).fillna("unknown"),
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
    })
    return _cast(df, config.NEURON_SCHEMA)


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
                  fetch_neurons, validate_neurons, force_refresh)


def load_edges(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "edges", RELEASE, SYN_SOURCE, min_syn=config.FETCH_MIN_SYN),
                  fetch_edges, validate_edges, force_refresh)


def load_edgeroi(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "edgeroi", RELEASE, SYN_SOURCE, min_syn=config.FETCH_MIN_SYN),
                  fetch_edgeroi, validate_edgeroi, force_refresh)

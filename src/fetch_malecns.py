"""MaleCNS (male-cns v1.0) fetch functions: neurons, edges, edgeroi.

Sources are Janelia's public bulk "flat connectome" files for v1.0 (src/paths.py),
so no neuPrint token is needed. They are already filtered at synapse confidence
0.5 (config.MALECNS_SYN_CONFIDENCE). Only neurons with status Traced
(config.MALECNS_STATUS) are kept, and edges between them.

Raw files are downloaded once into data/raw/; the tables built from them go
through the cache helper. Nothing downstream sees Janelia's column names.

Region names are kept in neuPrint's spelling (ME(R), CA(L), ...); eda_shared's
canonical_neuropil maps them to FlyWire base names.
"""
from __future__ import annotations

import logging
import shutil
import urllib.request

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

import config
from src.cache import cached
from src.fetch_fafb import _cast, recode
from src.paths import MALECNS_FILES, MALECNS_RAW, cache_file
from src.validate import validate_edgeroi, validate_edges, validate_neurons

log = logging.getLogger(__name__)

DATASET = "malecns"
RELEASE = config.MALECNS_DATASET.split(":")[1]                      # v1.0
SOURCE = config.MALECNS_STATUS.lower()                               # traced
CONF = config.MALECNS_SYN_CONFIDENCE                                 # 0.5
UNASSIGNED = "UNASGD"      # FlyWire's spelling for "no neuropil"; canonical_neuropil maps it to unknown
UNSPECIFIED = "unspecified"  # in every MaleCNS label for synapses outside a named neuropil

# Source value -> agreed vocabulary. An unmapped value raises (recode).
SIDE_MAP = {"L": "left", "R": "right", "M": "center", "unknown": "unknown"}
NT_MAP = {v: v for v in ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin",
                         "octopamine", "histamine"]}
NT_MAP["unclear"] = None                                             # -> unknown
# PROVISIONAL, to raise with the group: config.SUPER_CLASS_VALUES has no
# MaleCNS-only levels yet ("after EDA"), so every MaleCNS superclass is mapped
# onto the FAFB levels. Levels with no FAFB counterpart (nerve cord intrinsic,
# enteric, unconfirmed "_tbc" without a clear base) become unknown; the cell
# class and type columns keep the detail.
SUPER_CLASS_MAP = {
    "ol_intrinsic": "optic", "cb_intrinsic": "central",
    "visual_projection": "visual_projection", "visual_projection_tbc": "visual_projection",
    "visual_centrifugal": "visual_centrifugal",
    "ol_sensory": "sensory", "cb_sensory": "sensory", "vnc_sensory": "sensory",
    "cb_sensory_tbc": "sensory", "vnc_sensory_tbc": "sensory",
    "sensory_ascending": "sensory", "sensory_ascending_tbc": "sensory", "sensory_descending": "sensory",
    "ascending_neuron": "ascending", "descending_neuron": "descending",
    "cb_motor": "motor", "vnc_motor": "motor",
    "cb_efferent": "motor", "vnc_efferent": "motor", "efferent_ascending": "motor",
    "efferent_descending": "motor",
    "cb_endocrine": "endocrine", "vnc_endocrine": "endocrine",
    "vnc_intrinsic": None, "ENS": None, "vnc_tbc": None,
}


def raw_file(key: str):
    """Path to a raw MaleCNS file, downloading it on first use."""
    name, url = MALECNS_FILES[key]
    path = MALECNS_RAW / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", name)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url.format(name=name)) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(path)
    return path


def traced_ids() -> np.ndarray:
    ann = pd.read_feather(raw_file("annotations"), columns=["bodyId", "status"])
    return np.sort(ann.loc[ann["status"] == config.MALECNS_STATUS, "bodyId"].to_numpy("int64"))


def _partners(columns: list[str]) -> pa.Table:
    return pf.read_table(raw_file("syn_partners"), columns=columns)


def _region_from_all_synapses(ids: np.ndarray) -> pd.Series:
    """Neuropil holding most of each neuron's synapses (inputs plus outputs).

    Counted from the synapse-partner table over all partners, Traced or not
    ("from all synapses"), one count per synaptic connection on each side, as
    FlyWire's per-neuron neuropil counts do for FAFB. The neuropil of a
    connection is the table's primary_post. Synapses outside every neuropil
    (labelled <unspecified>, CentralBrain-unspecified, Optic-unspecified(L), ...)
    never win. Ties: the alphabetically first neuropil, until week 3 decides.
    """
    t = _partners(["body_pre", "body_post", "primary_post"])
    t = t.set_column(2, "primary_post", pc.cast(t["primary_post"], pa.string()))
    keep = pa.array(ids)
    parts = []
    for side in ("body_pre", "body_post"):
        s = t.select([side, "primary_post"]).filter(pc.is_in(t[side], value_set=keep))
        s = s.filter(pc.and_(pc.is_valid(s["primary_post"]),
                             pc.invert(pc.match_substring(s["primary_post"], UNSPECIFIED))))
        g = s.group_by([side, "primary_post"]).aggregate([([], "count_all")])
        parts.append(g.rename_columns(["neuron_id", "neuropil", "count"]).to_pandas())
    counts = pd.concat(parts).groupby(["neuron_id", "neuropil"], observed=True)["count"].sum().reset_index()
    counts = counts.sort_values(["neuron_id", "count", "neuropil"], ascending=[True, False, True])
    top = counts.drop_duplicates("neuron_id")
    second = counts[counts.duplicated("neuron_id")].drop_duplicates("neuron_id").set_index("neuron_id")["count"]
    ties = int((top.set_index("neuron_id")["count"] == second.reindex(top["neuron_id"]).values).sum())
    log.info("region: %d neurons tie between their top two neuropils", ties)
    return top.set_index("neuron_id")["neuropil"].astype(str)


def fetch_neurons() -> pd.DataFrame:
    ann = pd.read_feather(raw_file("annotations"))
    ann = ann[ann["status"] == config.MALECNS_STATUS].reset_index(drop=True)
    ids = ann["bodyId"].to_numpy("int64")

    nt = pd.read_feather(raw_file("neurotransmitters"),
                         columns=["body", "predicted_nt", "predicted_nt_confidence", "ground_truth"]).set_index("body")
    stats = pd.read_feather(raw_file("body_stats"), columns=["body", "pre", "post"])
    stats = stats[stats["body"].isin(ids)].set_index("body")

    side_src = ann["somaSide"].where(ann["somaSide"].notna(), ann["rootSide"])     # somaSide, else rootSide
    nt_type = recode(ann["bodyId"].map(nt["predicted_nt"]), NT_MAP, "nt_type")
    df = pd.DataFrame({
        "neuron_id": ann["bodyId"].astype("int64"),
        "dataset": DATASET,
        "release": RELEASE,
        "cell_type": ann["type"],
        "cell_class": ann["class"],
        "cell_sub_class": ann["subclass"],
        "super_class": recode(ann["superclass"], SUPER_CLASS_MAP, "super_class"),
        "side": recode(side_src, SIDE_MAP, "side"),
        "region": ann["bodyId"].map(_region_from_all_synapses(ids)).fillna("unknown"),
        "nt_type": nt_type,
        "nt_confidence": ann["bodyId"].map(nt["predicted_nt_confidence"]).where(nt_type != "unknown"),
        "n_pre": ann["bodyId"].map(stats["pre"]),
        "n_post": ann["bodyId"].map(stats["post"]),
        "flywire_type": ann["flywireType"],
        "manc_type": ann["mancType"],
        "hemibrain_type": ann["hemibrainType"],
        "dimorphism": ann["dimorphism"],
        "status": ann["status"],
        "hemilineage": ann["itoleeHl"],
        "nt_known": ann["bodyId"].map(nt["ground_truth"]),          # verbatim, as FAFB known_nt
    })
    return _cast(df, config.NEURON_SCHEMA)


def _finish(e: pd.DataFrame) -> pd.DataFrame:
    self_conn = e["pre_id"] == e["post_id"]
    log.info("dropping %d self-connections", int(self_conn.sum()))
    e = e[~self_conn]
    e = e[e["syn_count"] >= config.FETCH_MIN_SYN]
    e["dataset"] = DATASET
    e["release"] = RELEASE
    e["min_syn_fetched"] = config.FETCH_MIN_SYN
    return e


def fetch_edges() -> pd.DataFrame:
    """One row per pair of Traced neurons; Janelia's weights are already summed over neuropils."""
    t = pf.read_table(raw_file("weights"), columns=["body_pre", "body_post", "weight"])
    keep = pa.array(traced_ids())
    t = t.filter(pc.and_(pc.is_in(t["body_pre"], value_set=keep), pc.is_in(t["body_post"], value_set=keep)))
    e = t.rename_columns(["pre_id", "post_id", "syn_count"]).to_pandas()
    return _cast(_finish(e), config.EDGE_SCHEMA)


def fetch_edgeroi() -> pd.DataFrame:
    """One row per (pair, neuropil) of Traced neurons, counted from the synapse-partner table.
    Every connection is kept, so per-pair sums equal the edges table. Those outside a
    named neuropil keep MaleCNS's labels (<unspecified>, CentralBrain-unspecified, ...);
    a missing label would become UNASGD, though v1.0 has none."""
    t = _partners(["body_pre", "body_post", "primary_post"])
    keep = pa.array(traced_ids())
    t = t.filter(pc.and_(pc.is_in(t["body_pre"], value_set=keep), pc.is_in(t["body_post"], value_set=keep)))
    t = t.set_column(2, "primary_post", pc.fill_null(pc.cast(t["primary_post"], pa.string()), UNASSIGNED))
    g = t.group_by(["body_pre", "body_post", "primary_post"]).aggregate([([], "count_all")])
    e = g.rename_columns(["pre_id", "post_id", "neuropil", "syn_count"]).to_pandas()
    return _cast(_finish(e), config.EDGEROI_SCHEMA)


def load_neurons(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "neurons", RELEASE, SOURCE),
                  fetch_neurons, validate_neurons, force_refresh)


def load_edges(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "edges", RELEASE, SOURCE, min_syn=config.FETCH_MIN_SYN, conf=CONF),
                  fetch_edges, validate_edges, force_refresh)


def load_edgeroi(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "edgeroi", RELEASE, SOURCE, min_syn=config.FETCH_MIN_SYN, conf=CONF),
                  fetch_edgeroi, validate_edgeroi, force_refresh)

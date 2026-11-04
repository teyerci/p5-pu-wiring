"""MaleCNS (male-cns v1.0) fetch functions: neurons, edges, edgeroi.

Sources are Janelia's public bulk "flat connectome" files for v1.0 (src/paths.py), plus
neuPrint's per-ROI counts for region, which need a token
(scripts/fetch_malecns_roi_counts.py). The bulk files are already filtered at synapse confidence
0.5 (config.MALECNS_SYN_CONFIDENCE). Only neurons with status Traced
(config.MALECNS_STATUS) are kept, and edges between them.

Raw files are downloaded once into data/raw/; the tables built from them go
through the cache helper. Nothing downstream sees Janelia's column names.
load_brain builds the brain part (Corrections 4), the table to use beside fafb.

Region names are kept in neuPrint's spelling (ME(R), CA(L), ...); eda_shared's
canonical_neuropil maps them to FlyWire base names.
"""
from __future__ import annotations

import json
import logging
import shutil
import urllib.request

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

import config
from eda_shared.corrections import (add_nt_type_corrected, malecns_brain_part, malecns_neuropil_counts,
                                    map_super_class, region_from_counts)
from eda_shared.eda_helpers import MALECNS_NERVE_CORD
from src.cache import cached
from src.fetch_fafb import _cast, recode
from src.paths import MALECNS_FILES, MALECNS_RAW, MALECNS_ROI_COUNTS, cache_file
from src.validate import validate_dataset, validate_edgeroi, validate_edges, validate_neurons

log = logging.getLogger(__name__)

DATASET = "malecns"
RELEASE = config.MALECNS_DATASET.split(":")[1]                      # v1.0
SOURCE = config.MALECNS_STATUS.lower()                               # traced
BRAIN_SOURCE = f"{SOURCE}-brain"
CONF = config.MALECNS_SYN_CONFIDENCE                                 # 0.5
UNASSIGNED = "UNASGD"      # FlyWire's spelling for "no neuropil"; canonical_neuropil maps it to unknown

# Source value -> agreed vocabulary. An unmapped value raises (recode).
SIDE_MAP = {"L": "left", "R": "right", "M": "center", "unknown": "unknown"}
NT_MAP = {v: v for v in ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin",
                         "octopamine", "histamine"]}
NT_MAP["unclear"] = None                                             # -> unknown
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


def _roi_counts() -> pd.DataFrame:
    """Per neuron and primary ROI, upstream + downstream over all partners (neuPrint)."""
    if not MALECNS_ROI_COUNTS.exists():
        raise FileNotFoundError(f"{MALECNS_ROI_COUNTS} is missing: run scripts/fetch_malecns_roi_counts.py")
    return malecns_neuropil_counts(pd.read_parquet(MALECNS_ROI_COUNTS))


def fetch_neurons() -> pd.DataFrame:
    """Every Traced neuron. Region follows the shared rule over brain neuropils, so a
    neuron with no connection in the brain has region unknown and is not in load_brain."""
    ann = pd.read_feather(raw_file("annotations"))
    ann = ann[ann["status"] == config.MALECNS_STATUS].reset_index(drop=True)
    ids = ann["bodyId"].to_numpy("int64")

    nt = pd.read_feather(raw_file("neurotransmitters"),
                         columns=["body", "predicted_nt", "predicted_nt_confidence", "ground_truth"]).set_index("body")
    stats = pd.read_feather(raw_file("body_stats"), columns=["body", "pre", "post"])
    stats = stats[stats["body"].isin(ids)].set_index("body")

    side_src = ann["somaSide"].where(ann["somaSide"].notna(), ann["rootSide"])     # somaSide, else rootSide
    nt_type = recode(ann["bodyId"].map(nt["predicted_nt"]), NT_MAP, "nt_type")
    sc = map_super_class(ann["superclass"], DATASET)                    # correction 2
    reg = region_from_counts(ids, _roi_counts(), nerve_cord=MALECNS_NERVE_CORD).set_index("neuron_id")   # correction 6
    df = pd.DataFrame({
        "neuron_id": ann["bodyId"].astype("int64"),
        "dataset": DATASET,
        "release": RELEASE,
        "cell_type": ann["type"],
        "cell_class": ann["class"],
        "cell_sub_class": ann["subclass"],
        "super_class": sc["super_class"].array,
        "side": recode(side_src, SIDE_MAP, "side"),
        "region": ann["bodyId"].map(reg["region"]),
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
        "super_class_source": sc["super_class_source"].array,
    })
    return _cast(add_nt_type_corrected(df), config.NEURON_SCHEMA)       # correction 1


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
                  fetch_neurons, validate_neurons, force_refresh, columns=config.NEURON_SCHEMA)


def load_edges(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "edges", RELEASE, SOURCE, min_syn=config.FETCH_MIN_SYN, conf=CONF),
                  fetch_edges, validate_edges, force_refresh)


def load_edgeroi(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "edgeroi", RELEASE, SOURCE, min_syn=config.FETCH_MIN_SYN, conf=CONF),
                  fetch_edgeroi, validate_edgeroi, force_refresh)


def load_brain(force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """The brain part (Corrections 4): neurons, edges, edgeroi and the Step 0 counts.

    Cropped by synapse location, so edge weights count brain synapses only. These
    edges are the denominator for relative weights (Corrections 5).
    """
    n_path = cache_file(DATASET, "neurons", RELEASE, BRAIN_SOURCE)
    e_path = cache_file(DATASET, "edges", RELEASE, BRAIN_SOURCE, min_syn=config.FETCH_MIN_SYN, conf=CONF)
    r_path = cache_file(DATASET, "edgeroi", RELEASE, BRAIN_SOURCE, min_syn=config.FETCH_MIN_SYN, conf=CONF)
    report_path = n_path.with_suffix(".report.json")
    paths = [n_path, e_path, r_path, report_path]
    if all(p.exists() for p in paths) and not force_refresh:
        return (pd.read_parquet(n_path), pd.read_parquet(e_path), pd.read_parquet(r_path),
                json.loads(report_path.read_text()))
    edgeroi = load_edgeroi().astype({"neuropil": "category"})           # eda_shared's memory advice
    n, e, er, report = malecns_brain_part(load_neurons(force_refresh), edgeroi, _roi_counts())
    n, e, er = _cast(n, config.NEURON_SCHEMA), _cast(e, config.EDGE_SCHEMA), _cast(er, config.EDGEROI_SCHEMA)
    validate_dataset(n, e, er)
    n_path.parent.mkdir(parents=True, exist_ok=True)
    n.to_parquet(n_path, index=False)
    e.to_parquet(e_path, index=False)
    er.to_parquet(r_path, index=False)
    report_path.write_text(json.dumps(report, indent=1))
    return n, e, er, report

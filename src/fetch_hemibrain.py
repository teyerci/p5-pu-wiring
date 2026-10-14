"""Hemibrain fetch functions: neurons, edges, edgeroi (for EDA item E6).

Source: Janelia's public export of the hemibrain **v1.2** traced adjacencies
(src/paths.py), which holds every non-cropped Traced neuron. The plan pins
v1.2.1, which is only in neuPrint (token needed); using v1.2 is a stated
deviation until a token is available. Cropped neurons are not in the export,
so the plan's "cropped hemibrain neurons removed" holds by construction.

The export has no side, class, neurotransmitter or synapse-total columns.
Side: the last _R / _L token of the instance (EPG(PB08)_R4, PFNv(PB05)_R4_C7,
Delta7(PB15)_L4R5_R -> right; for central-complex columnar types this token is
the protocerebral-bridge half, assumed to be the soma side); without a token,
the side of the neuron's region (LO(R) -> right); otherwise unknown. Region: the
primary ROI holding most of the neuron's synapses with partners in the export
(non-cropped Traced neurons), NotPrimary ignored. Everything else is left
unknown or missing.
"""
from __future__ import annotations

import io
import logging
import re
import shutil
import tarfile
import urllib.request

import pandas as pd

from src.cache import cached
from src.fetch_fafb import _cast
from src.paths import HEMIBRAIN_FILES, HEMIBRAIN_RAW, cache_file
from src.validate import validate_edgeroi, validate_edges, validate_neurons

import config

log = logging.getLogger(__name__)

DATASET = "hemibrain"
RELEASE = "v1.2"
SOURCE = "traced-uncropped"
NOT_PRIMARY = "NotPrimary"
UNASSIGNED = "UNASGD"
STATUS = "Traced"                       # every exported neuron is Traced
SIDE_FROM_SUFFIX = {"R": "right", "L": "left"}
_SIDE_TOKEN = re.compile(r"_([LR])(?=\d|_|$)")


def side_of(instance: pd.Series, region: pd.Series) -> pd.Series:
    """Instance side token (the last one), else the region's (L)/(R) suffix, else unknown."""
    token = instance.fillna("").map(lambda s: (_SIDE_TOKEN.findall(s) or [None])[-1])
    from_region = region.astype("string").str.extract(r"\(([LR])\)$")[0]
    side = token.where(token.notna(), from_region)
    return side.map(SIDE_FROM_SUFFIX).fillna("unknown")


def raw_file(key: str):
    """Path to a raw hemibrain archive, downloading it on first use."""
    name, url = HEMIBRAIN_FILES[key]
    path = HEMIBRAIN_RAW / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", name)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url.format(name=name), timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(path)
    return path


def read_member(key: str, member: str) -> pd.DataFrame:
    """Read one CSV from a raw archive without unpacking it to disk."""
    with tarfile.open(raw_file(key), "r:gz") as tar:
        name = next(m for m in tar.getnames() if m.endswith("/" + member))
        return pd.read_csv(io.BytesIO(tar.extractfile(name).read()))


def _region(roi: pd.DataFrame) -> pd.Series:
    """Primary ROI with most synapses per neuron (inputs plus outputs); ties -> alphabetically first."""
    roi = roi[roi["roi"] != NOT_PRIMARY]
    parts = [roi.groupby([side, "roi"])["weight"].sum().rename_axis(["neuron_id", "roi"])
             for side in ("bodyId_pre", "bodyId_post")]
    counts = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
    counts = counts.sort_values(["neuron_id", "weight", "roi"], ascending=[True, False, True])
    return counts.drop_duplicates("neuron_id").set_index("neuron_id")["roi"]


def fetch_neurons() -> pd.DataFrame:
    tn = read_member("uncropped", "traced-neurons.csv")
    region = _region(read_member("uncropped", "traced-roi-connections.csv"))
    reg = tn["bodyId"].map(region).fillna("unknown")
    df = pd.DataFrame({
        "neuron_id": tn["bodyId"].astype("int64"),
        "dataset": DATASET,
        "release": RELEASE,
        "cell_type": tn["type"],
        "cell_class": pd.NA, "cell_sub_class": pd.NA,
        "super_class": "unknown",
        "side": side_of(tn["instance"], reg),
        "region": reg,
        "nt_type": "unknown", "nt_confidence": float("nan"),
        "n_pre": pd.NA, "n_post": pd.NA,
        "flywire_type": pd.NA, "manc_type": pd.NA,
        "hemibrain_type": tn["type"],
        "dimorphism": pd.NA,
        "status": STATUS,
        "hemilineage": pd.NA, "nt_known": pd.NA,
    })
    return _cast(df, config.NEURON_SCHEMA)


def _finish(e: pd.DataFrame) -> pd.DataFrame:
    self_conn = e["pre_id"] == e["post_id"]
    log.info("dropping %d self-connections", int(self_conn.sum()))
    e = e[~self_conn]
    e = e[e["syn_count"] >= config.FETCH_MIN_SYN].copy()
    e["dataset"], e["release"], e["min_syn_fetched"] = DATASET, RELEASE, config.FETCH_MIN_SYN
    return e


def fetch_edges() -> pd.DataFrame:
    t = read_member("uncropped", "traced-total-connections.csv")
    e = t.rename(columns={"bodyId_pre": "pre_id", "bodyId_post": "post_id", "weight": "syn_count"})
    return _cast(_finish(e), config.EDGE_SCHEMA)


def fetch_edgeroi() -> pd.DataFrame:
    """Per (pair, primary ROI); synapses outside every primary ROI (NotPrimary) are kept as UNASGD,
    so per-pair sums equal the edges table."""
    t = read_member("uncropped", "traced-roi-connections.csv")
    t["roi"] = t["roi"].replace(NOT_PRIMARY, UNASSIGNED)
    e = t.rename(columns={"bodyId_pre": "pre_id", "bodyId_post": "post_id", "roi": "neuropil", "weight": "syn_count"})
    return _cast(_finish(e), config.EDGEROI_SCHEMA)


def cropped_typed_share(types=None) -> dict:
    """Estimate of the cropped share among typed neurons: typed bodies in Janelia's
    all-traced export that are missing from the non-cropped export. The all-traced
    export also carries other statuses, so this is an upper bound; neuPrint's
    `cropped` flag (v1.2.1, token) would give the exact figure."""
    allt = read_member("all_traced", "neurons.csv")
    uncropped = set(read_member("uncropped", "traced-neurons.csv")["bodyId"])
    typed = allt[allt["type"].notna()]
    if types is not None:
        typed = typed[typed["type"].isin(types)]
    missing = ~typed["bodyId"].isin(uncropped)
    return {"typed_neurons": int(len(typed)), "not_in_uncropped": int(missing.sum()),
            "share": float(missing.mean()) if len(typed) else float("nan")}


def load_neurons(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "neurons", RELEASE, SOURCE), fetch_neurons, validate_neurons, force_refresh)


def load_edges(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "edges", RELEASE, SOURCE, min_syn=config.FETCH_MIN_SYN),
                  fetch_edges, validate_edges, force_refresh)


def load_edgeroi(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_file(DATASET, "edgeroi", RELEASE, SOURCE, min_syn=config.FETCH_MIN_SYN),
                  fetch_edgeroi, validate_edgeroi, force_refresh)

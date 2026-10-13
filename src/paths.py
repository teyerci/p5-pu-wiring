"""Repository paths and raw-download addresses.

config.py is the course's shared file and stays unchanged, so anything
project-local (raw-file locations, download URLs) lives here instead.
"""
from __future__ import annotations

from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / config.CACHE_DIR          # config.CACHE_DIR is relative to the repo root
RAW_DIR = ROOT / "data" / "raw"
FAFB_RAW = RAW_DIR / f"fafb_v{config.FAFB_MATERIALIZATION}"

# FAFB 783 bulk files. The connection table is the Buhmann synapse table
# aggregated per (pre, post, neuropil); Zenodo record 10676866, version 783.0.
_ZENODO = "https://zenodo.org/api/records/10676866/files/{name}/content"
_ANNOTATIONS = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
                f"{config.FAFB_ANNOTATIONS_TAG}/supplemental_files/{{name}}")

FAFB_FILES = {
    "connections": ("proofread_connections_783.feather", _ZENODO),
    "neuropil_pre": ("per_neuron_neuropil_count_pre_783.feather", _ZENODO),
    "neuropil_post": ("per_neuron_neuropil_count_post_783.feather", _ZENODO),
    "proofread_ids": ("proofread_root_ids_783.npy", _ZENODO),
    "annotations": ("Supplemental_file1_neuron_annotations.tsv", _ANNOTATIONS),
}


def cache_file(*args, **kwargs) -> Path:
    """config.cache_path, resolved against the repository root."""
    return ROOT / config.cache_path(*args, **kwargs)

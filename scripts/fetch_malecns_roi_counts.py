"""Download neuPrint's per-ROI counts for every Traced MaleCNS neuron, primary ROIs only.

Writes data/raw/malecns_v1.0/neuprint_roi_counts_primary.parquet (bodyId, roi, pre,
post, upstream, downstream), the input of the shared region rule (Corrections 4 and 6).
Needs NEUPRINT_APPLICATION_CREDENTIALS, from the environment or the gitignored .env.
Batches already written are skipped, so an interrupted run can be restarted.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))

import neuprint  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.fetch_malecns import traced_ids  # noqa: E402
from src.paths import MALECNS_RAW, MALECNS_ROI_COUNTS  # noqa: E402

BATCH = 5000
OUT = MALECNS_ROI_COUNTS
PARTS = MALECNS_RAW / "neuprint_roi_counts_parts"


def token() -> str:
    if os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS"):
        return os.environ["NEUPRINT_APPLICATION_CREDENTIALS"]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("NEUPRINT_APPLICATION_CREDENTIALS="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no NEUPRINT_APPLICATION_CREDENTIALS in the environment or .env")


def main() -> None:
    client = neuprint.Client("neuprint.janelia.org", dataset=config.MALECNS_DATASET, token=token())
    primary = set(neuprint.fetch_primary_rois(client=client))
    ids = traced_ids()
    PARTS.mkdir(parents=True, exist_ok=True)
    batches = np.array_split(ids, int(np.ceil(len(ids) / BATCH)))
    for i, batch in enumerate(batches):
        part = PARTS / f"{i:04d}.parquet"
        if part.exists():
            continue
        _, roi = neuprint.fetch_neurons(neuprint.NeuronCriteria(bodyId=batch.tolist(), client=client),
                                        client=client)
        roi = roi[roi["roi"].isin(primary)]
        roi = roi[["bodyId", "roi", "pre", "post", "upstream", "downstream"]]
        roi.astype({"roi": "string"}).to_parquet(part, index=False)
        print(f"{i + 1}/{len(batches)}", flush=True)
    all_rows = pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob("*.parquet"))], ignore_index=True)
    all_rows.to_parquet(OUT, index=False)
    print(f"{len(all_rows):,} rows, {all_rows['bodyId'].nunique():,} neurons -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

"""Realised c under synapse and connection dropout on the fafb central-complex roster
(τ = 10, q ∈ {1, 0.9, 0.7}, 5 seeds). Writes reports/dropout_realised_c.csv."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import fetch_fafb                               # noqa: E402
from src.dropout import realised_c                       # noqa: E402
from src.pairs import cx_roster, region_pair_counts      # noqa: E402

neurons, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edgeroi()
pair_syn = region_pair_counts(edgeroi, cx_roster(neurons))
tables = [realised_c(pair_syn, q, mode=m) for q in (1.0, 0.9, 0.7) for m in ("synapse", "connection")]
out = pd.concat(tables).rename_axis("stratum").reset_index()
out.to_csv(ROOT / "reports" / "dropout_realised_c.csv", index=False)
print(out.round(3).to_string())

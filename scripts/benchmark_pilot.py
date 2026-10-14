"""First run of the dropout benchmark (one seed, one split): fafb central-complex roster,
q ∈ {1, 0.9, 0.7}, synapse and connection dropout. Writes reports/benchmark_pilot.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import fetch_fafb                       # noqa: E402
from src.benchmark import run_condition          # noqa: E402
from src.pairs import cx_roster                  # noqa: E402

neurons, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edgeroi()
ids = cx_roster(neurons)
er = edgeroi[edgeroi["pre_id"].isin(ids) | edgeroi["post_id"].isin(ids)].reset_index(drop=True)
results = []
for q, mode in [(1.0, "synapse"), (0.9, "synapse"), (0.7, "synapse"), (0.9, "connection"), (0.7, "connection")]:
    r = run_condition(neurons, er, ids, q, mode, seed=0)
    results.append(r)
    print(json.dumps(r, default=float), flush=True)
(ROOT / "reports" / "benchmark_pilot.json").write_text(json.dumps(results, indent=1, default=float))

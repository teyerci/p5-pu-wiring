"""λ sensitivity of the benchmark (diagnostic, not a selection): q = 0.7, seeds 0-2,
λ ∈ {1e-2, 1e-3, 1e-4}, for the dropout mode given on the command line (synapse by
default). Writes reports/benchmark_lambda[_<mode>].json. The final design chooses λ by
inner cross-validation on training pairs, never on the held-out scores."""
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
MODE = sys.argv[1] if len(sys.argv) > 1 else "synapse"
OUT = ROOT / "reports" / ("benchmark_lambda.json" if MODE == "synapse" else f"benchmark_lambda_{MODE}.json")
results = []
for lam in (1e-2, 1e-3, 1e-4):
    for seed in range(3):
        r = run_condition(neurons, er, ids, 0.7, MODE, seed=seed, lam=lam)
        r["lambda"] = lam
        results.append(r)
        print(lam, seed, {k: (round(v["brier_y"], 6), round(v["auc_y"], 4)) for k, v in r["models"].items() if k != "pu_e1"}, flush=True)
        OUT.write_text(json.dumps(results, indent=1, default=float))

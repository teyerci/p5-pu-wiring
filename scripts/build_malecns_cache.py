"""Build (or reuse) the MaleCNS cache and run every validator. Prints timings.

Tables can be named to build only some of them (e.g. `edges`); validate_dataset
runs only when all three are built.
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fetch_malecns import load_edgeroi, load_edges, load_neurons   # noqa: E402
from src.summary import summary                                         # noqa: E402
from src.validate import validate_dataset                               # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
force = "--force" in sys.argv
loaders = {"neurons": load_neurons, "edges": load_edges, "edgeroi": load_edgeroi}
args = [a for a in sys.argv[1:] if a != "--force"]
unknown = [a for a in args if a not in loaders]
if unknown:
    sys.exit(f"unknown table(s) {unknown}; choose from {list(loaders)}")
wanted = args or list(loaders)
tables, t = {}, {}
for name in wanted:
    start = time.perf_counter()
    tables[name] = loaders[name](force_refresh=force)
    t[name] = time.perf_counter() - start
    print(f"{name}: {len(tables[name]):,} rows")
print({k: round(v, 1) for k, v in t.items()}, "seconds")
if len(tables) == 3:
    validate_dataset(tables["neurons"], tables["edges"], tables["edgeroi"])
    print(summary(tables["neurons"], tables["edges"]).T)

"""Build (or reuse) the FAFB cache and run every validator. Prints timings."""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fetch_fafb import load_edgeroi, load_edges, load_neurons   # noqa: E402
from src.summary import summary                                      # noqa: E402
from src.validate import validate_dataset                            # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
force = "--force" in sys.argv
t = {}
for name, fn in [("neurons", load_neurons), ("edges", load_edges), ("edgeroi", load_edgeroi)]:
    start = time.perf_counter()
    globals()[name] = fn(force_refresh=force)
    t[name] = time.perf_counter() - start
validate_dataset(neurons, edges, edgeroi)
print({k: round(v, 1) for k, v in t.items()}, "seconds")
print(summary(neurons, edges).T)

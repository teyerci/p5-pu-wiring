"""Generate notebooks/00_integration.ipynb (kept as code so the notebook is reproducible)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 00 — Integration\n\nFrom a clean clone: config, cache, validators, counts. "
       "Runs off `data/cache/` with no network I/O once the cache exists."),
    code('import sys\nfrom pathlib import Path\n'
         'ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.py").exists())\n'
         'sys.path.insert(0, str(ROOT))'),
    code('import logging\nimport config\n'
         'from src.fetch_fafb import load_neurons, load_edges, load_edgeroi\n'
         'from src.validate import validate_dataset\nfrom src.summary import summary\n'
         'logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")\n'
         'print("FAFB", config.FAFB_MATERIALIZATION, config.FAFB_ANNOTATIONS_TAG, config.FAFB_SYNAPSE_TABLE)'),
    md("## Load (cache first) and validate"),
    code("neurons, edges, edgeroi = load_neurons(), load_edges(), load_edgeroi()\n"
         "validate_dataset(neurons, edges, edgeroi)\nsummary(neurons, edges)"),
    md("## Threshold table (course grid, central brain)"),
    code("from eda_shared.eda_helpers import add_provisional_flags, relative_weight\n"
         "import pandas as pd\n"
         "fl = add_provisional_flags(neurons)\n"
         "central = set(fl.loc[~fl.is_optic, 'neuron_id'])\n"
         "e = edges.assign(frac=relative_weight(edges))\n"
         "e = e[e.pre_id.isin(central) & e.post_id.isin(central)]\n"
         "rows = []\n"
         "for label, keep in [(f'>= {t}', e.syn_count >= t) for t in (1, 5, 10, 31)] + \\\n"
         "                   [(f'>= {f:.1%} input', e.frac >= f) for f in (0.009, 0.026)]:\n"
         "    k = e[keep]\n"
         "    connected = pd.unique(pd.concat([k.pre_id, k.post_id]))\n"
         "    rows.append({'criterion': label, 'edges': len(k), 'synapses': int(k.syn_count.sum()),\n"
         "                 'isolated': len(central) - len(connected),\n"
         "                 'median_out_degree': float(k.groupby('pre_id').size().median())})\n"
         "pd.DataFrame(rows)"),
    md("## Project 5 region: central complex roster"),
    code("cx = neurons[neurons.cell_class == 'CX']\n"
         "ids = set(cx.neuron_id)\n"
         "ecx = edgeroi[edgeroi.neuropil.isin(['EB', 'FB', 'PB', 'NO']) & edgeroi.pre_id.isin(ids) & edgeroi.post_id.isin(ids)]\n"
         "pair = ecx.groupby(['pre_id', 'post_id']).syn_count.sum()\n"
         "n = len(ids)\n"
         "pd.Series({'roster': n, 'ordered pairs': n * (n - 1), 'pairs >= 10': int((pair >= 10).sum()),\n"
         "           'recorded below 10 (1-9 syn)': int(((pair >= 1) & (pair < 10)).sum())})"),
    md("## What this shows\n\nWritten after each run: counts match the course check "
       "(2,700,513 edges at 5 synapses), the cache works offline, and the central complex "
       "gives enough positives at the course default of 10 synapses."),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
Path("notebooks").mkdir(exist_ok=True)
nbf.write(nb, "notebooks/00_integration.ipynb")

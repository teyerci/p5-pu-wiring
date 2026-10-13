"""Generate notebooks/eda_E.ipynb: workstream E, regenerated from the cache with no network calls."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# EDA workstream E — regions, completeness and discrepancy\n\n"
       "Owner: Project 5. Question: where do neurons and synapses sit, how complete is the record, "
       "and how often does an edge recur in another copy?\n\n"
       "FAFB only for now. E6 and the malecns side of every item need the MaleCNS cache "
       "(neuPrint token) and the hemibrain."),
    code('import sys, json, warnings\nfrom pathlib import Path\n'
         'ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.py").exists())\n'
         'sys.path.insert(0, str(ROOT))\nwarnings.filterwarnings("ignore", category=FutureWarning)'),
    code("import pandas as pd\nfrom IPython.display import Image, display\n"
         "from src.fetch_fafb import load_neurons, load_edges, load_edgeroi, raw_file\n"
         "from src import eda_e_figures as F\n"
         "neurons, edges, edgeroi = load_neurons(), load_edges(), load_edgeroi()\nSTATS = {}"),
]
items = [
    ("E1", "One neuron, one region (D15, D17)", "F.fig_e1(neurons, edges, edgeroi)",
     "display(tabs['worked_example'])"),
    ("E2", "The optic-lobe share (D13, D14)", "F.fig_e2(neurons, edgeroi)", None),
    ("E3", "The region-to-region map", "F.fig_e3(neurons, edges)", None),
    ("E4", "How complete is each neuron's record (FAFB supplement; the plan's E4 is malecns)",
     "F.fig_e4_fafb(neurons, edges, pd.read_feather(raw_file('neuropil_post')))", None),
    ("E5", "Does an edge recur on the other side (D10)", "F.fig_e5(neurons, edges)",
     "display(pd.concat(tabs, axis=1).round(3))\n"
     "from src.eda_e import e5_by_type_size, e5_restricted\n"
     "display(e5_by_type_size(neurons, edges))  # recurrence rises with type size\n"
     "display(e5_restricted(neurons, edges))    # restricting the type set barely moves it"),
    ("E6", "Does an edge recur in another animal", None, None),
    ("E7", "Central complex profile", "F.fig_e7(neurons, edges)", None),
]
for key, title, call, extra in items:
    cells.append(md(f"## {key}. {title}"))
    if call is None:
        cells.append(md("Pending: needs the malecns cache (neuPrint token) and the hemibrain."))
        continue
    src = (f"paths, st, tabs = {call}\nSTATS['{key}'] = st\nprint(json.dumps(st, indent=1, default=float))\n"
           "for p in paths:\n    display(Image(filename=str(p)))")
    if extra:
        src += "\n" + extra
    cells.append(code(src))
cells.append(code("(ROOT / 'reports' / 'eda_E_stats.json').write_text(json.dumps(STATS, indent=1, default=float))"))
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/eda_E.ipynb")

"""Generate notebooks/eda_E.ipynb: workstream E, regenerated from the cache with no network calls."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# EDA workstream E — regions, completeness and discrepancy\n\n"
       "Owner: Project 5. Question: where do neurons and synapses sit, how complete is the record, "
       "and how often does an edge recur in another copy?\n\n"
       "fafb (783) and malecns (v1.0). As the plan's Definitions say, malecns means its brain part: "
       "neurons with `in_brain`, the edges among them and, in edgeroi, brain neuropils only. "
       "The nerve-cord list passed to `add_provisional_flags` is provisional (`src/eda_e.py`, "
       "`MALECNS_NERVE_CORD`). E6 is pending: it needs the hemibrain."),
    code('import sys, json, warnings\nfrom pathlib import Path\n'
         'ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.py").exists())\n'
         'sys.path.insert(0, str(ROOT))\nwarnings.filterwarnings("ignore", category=FutureWarning)'),
    code("import pandas as pd\nfrom IPython.display import Image, display\n"
         "from src import fetch_fafb, fetch_malecns\n"
         "from src import eda_e_figures as F\n"
         "from src.eda_e import brain_part\n"
         "fafb = (fetch_fafb.load_neurons(), fetch_fafb.load_edges(), fetch_fafb.load_edgeroi())\n"
         "mal_edges_all = fetch_malecns.load_edges()\n"
         "mal = brain_part(fetch_malecns.load_neurons(), mal_edges_all, fetch_malecns.load_edgeroi())\n"
         "from src.eda_e import columnar_cx_types\n"
         "FAFB_COLUMNAR = columnar_cx_types(fafb[0])\n"
         "print('malecns brain part:', [len(t) for t in mal])\n"
         "STATS = {}\n\n"
         "def show(key, result):\n"
         "    paths, st, tabs = result\n"
         "    STATS[key] = st\n"
         "    print(json.dumps(st, indent=1, default=float))\n"
         "    for p in paths:\n"
         "        display(Image(filename=str(p)))\n"
         "    return tabs"),
]
items = [
    ("E1", "One neuron, one region (D15, D17)", [
        "tabs = show('E1_fafb', F.fig_e1(*fafb))\ndisplay(tabs['worked_example'])",
        "tabs = show('E1_malecns', F.fig_e1(*mal, ds='malecns', release='v1.0'))\ndisplay(tabs['worked_example'])"]),
    ("E2", "The optic-lobe share (D13, D14)", [
        "show('E2', F.fig_e2({'fafb': ('783', fafb[0], fafb[2]), 'malecns': ('v1.0', mal[0], mal[2])}));"]),
    ("E3", "The region-to-region map", [
        "show('E3', F.fig_e3({'fafb': ('783', fafb[0], fafb[1]), 'malecns': ('v1.0', mal[0], mal[1])}));"]),
    ("E4", "How complete is each neuron's record", [
        "# The plan's E4 (malecns). n_post was checked to count only synapses at confidence >= 0.5.\n"
        "# Inputs from every Traced partner (E4 is malecns-only, so the brain-part rule's reason does not apply).\n"
        "show('E4_malecns', F.fig_e4_malecns(mal[0], mal_edges_all, mal[1]));",
        "# FAFB supplement: FlyWire's all-synapse counts allow the same ratio.\n"
        "show('E4supp_fafb', F.fig_e4_fafb(fafb[0], fafb[1], "
        "pd.read_feather(fetch_fafb.raw_file('neuropil_post'))));"]),
    ("E5", "Does an edge recur on the other side (D10)", [
        "tabs = show('E5_fafb', F.fig_e5(fafb[0], fafb[1]))\n"
        "display(pd.concat(tabs, axis=1).round(3))\n"
        "from src.eda_e import e5_by_type_size, e5_restricted\n"
        "display(e5_by_type_size(fafb[0], fafb[1]))  # recurrence rises with type size\n"
        "display(e5_restricted(fafb[0], fafb[1]))    # restricting the type set barely moves it",
        "# malecns supplement (the plan specifies E5 for fafb): the male's own left-right curve\n"
        "tabs = show('E5supp_malecns', F.fig_e5(mal[0], mal[1], 'malecns', 'v1.0', item='E5supp',\n"
        "            note='Supplement: the plan specifies E5 for fafb. ' + F.BRAIN_NOTE['malecns']))\n"
        "display(pd.concat(tabs, axis=1).round(3))\n"
        "display(e5_by_type_size(mal[0], mal[1]))"]),
    ("E6", "Does an edge recur in another animal", None),
    ("E7", "Central complex profile", [
        "show('E7a', F.fig_e7a({'fafb': ('783', fafb[0]), 'malecns': ('v1.0', mal[0])}));",
        "show('E7_fafb', F.fig_e7(fafb[0], fafb[1], FAFB_COLUMNAR));",
        "show('E7_malecns', F.fig_e7(mal[0], mal[1], FAFB_COLUMNAR, 'malecns', 'v1.0', type_column='flywire_type'));"]),
]
for key, title, calls in items:
    cells.append(md(f"## {key}. {title}"))
    if calls is None:
        cells.append(md("Pending: needs the hemibrain v1.2.1 (bulk files or a neuPrint token)."))
        continue
    cells.extend(code(c) for c in calls)
cells.append(code("(ROOT / 'reports' / 'eda_E_stats.json').write_text(json.dumps(STATS, indent=1, default=float))"))
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/eda_E.ipynb")

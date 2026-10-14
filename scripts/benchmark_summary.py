"""Summarise reports/benchmark.json over seeds: a table (reports/benchmark_summary.csv)
and two figures (figures/eda/B1_..., B2_...)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt                                    # noqa: E402
from src.eda_plots import INK_2, MUTED, SERIES, finish, style      # noqa: E402

style()
runs = json.load(open(ROOT / "reports" / "benchmark.json"))
rows = []
for r in runs:
    for name, m in r["models"].items():
        rows.append({"condition": f"{r['mode']} q={r['q']}" if r["q"] < 1 else "no thinning", "q": r["q"],
                     "mode": r["mode"], "seed": r["seed"], "model": name, "c": m["c"], "realised_c": r["realised_c"],
                     "e1": r["e1"], "auc_y": m["auc_y"], "brier_y": m["brier_y"], "mean_p": m["mean_p"],
                     "mean_y": m["mean_y"], "recall_deleted_top1pct": m["recall_deleted_top1pct"],
                     "mean_f_deleted": m["mean_f_deleted"], "deleted_in_eval": r["deleted_in_eval"],
                     **{f"mean_f_true_{k}": v["mean_f"] for k, v in m["by_true_stratum"].items()}})
df = pd.DataFrame(rows)
metrics = ["c", "realised_c", "e1", "auc_y", "brier_y", "mean_p", "mean_y", "recall_deleted_top1pct",
           "mean_f_deleted", "deleted_in_eval"] + [c for c in df.columns if c.startswith("mean_f_true_")]
summary = df.groupby(["condition", "model"], sort=False)[metrics].agg(["mean", "std"])
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary["seeds"] = df.groupby(["condition", "model"], sort=False).size()
summary.reset_index().to_csv(ROOT / "reports" / "benchmark_summary.csv", index=False)
print(summary[["seeds", "realised_c_mean", "e1_mean", "auc_y_mean", "brier_y_mean", "brier_y_std", "mean_p_mean",
               "mean_y_mean", "recall_deleted_top1pct_mean", "mean_f_deleted_mean"]].round(5).to_string())

# B1: mean predicted P(y=1) vs truth, naive vs PU at the true c
conds = list(dict.fromkeys(df["condition"]))
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(conds))
for k, (model, label, col) in enumerate([("naive", "naive (c = 1)", SERIES[1]), ("pu_true_c", "PU at the true c", SERIES[0])]):
    g = summary.xs(model, level="model").reindex(conds)
    ax.bar(x + (k - 0.5) * 0.36, g["mean_p_mean"], width=0.34, yerr=g["mean_p_std"], color=col, label=label,
           error_kw={"ecolor": INK_2, "lw": 1})
truth = summary.xs("naive", level="model").reindex(conds)["mean_y_mean"]
for xi, t in zip(x, truth):
    ax.plot([xi - 0.4, xi + 0.4], [t, t], color=INK_2, lw=1.5, ls="--")
ax.set_xticks(x, conds)
ax.set(ylabel="Mean P(y=1) on held-out pairs", ylim=(0, None))
ax.legend(loc="upper right")
ax.text(x[-1] + 0.42, truth.iloc[-1], "truth", fontsize=8, color=INK_2, va="center")
finish(fig, "B1_fafb_783_cx_benchmark-level_min10",
       "At q = 0.7 the naive model under-predicts; PU at the true c lifts the level, overshooting by 7–26% across thinned conditions",
       "Dataset fafb 783; central-complex roster; truth = recorded pairs at τ = 10; models see a thinned copy (synapse "
       "or connection dropout at q). Held-out source types, split first; all deleted connections scored. Mean and sd "
       "over 5 seeds (1 without thinning); dashed line = true rate.")

# B2: mean P(y=1) for true positives by true synapse stratum, q = 0.7
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
strata = ["10–19", "20–49", "50+"]
for ax, cond in zip(axes, ["synapse q=0.7", "connection q=0.7"]):
    for k, (model, label, col) in enumerate([("naive", "naive", SERIES[1]), ("pu_true_c", "PU at true c", SERIES[0])]):
        g = summary.loc[(cond, model)]
        vals = [g.get(f"mean_f_true_{s}_mean", np.nan) for s in strata]
        errs = [g.get(f"mean_f_true_{s}_std", np.nan) for s in strata]
        ax.bar(np.arange(3) + (k - 0.5) * 0.36, vals, width=0.34, yerr=errs, color=col, label=label,
               error_kw={"ecolor": INK_2, "lw": 1})
    ax.set_xticks(range(3), strata)
    ax.set_title(f"{cond.split()[0]} dropout, q = 0.7", fontsize=10)
    ax.set(xlabel="True synapses of the pair")
axes[0].set(ylabel="Mean P(y=1), true connections")
axes[0].legend(loc="upper left")
finish(fig, "B2_fafb_783_cx_benchmark-strata_min10",
       "True connections get low probabilities in every stratum; PU at the true c lifts all strata about 2-fold",
       "Dataset fafb 783; central-complex roster; τ = 10; q = 0.7; held-out true connections (y = 1) by their true "
       "synapse count; mean and sd over 5 seeds.")

# B3: weighted reliability (mean prediction vs true rate per equal-weight bin), averaged over seeds
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, (q, mode) in zip(axes, [(0.7, "synapse"), (0.7, "connection")]):
    sel = [r for r in runs if r["q"] == q and r["mode"] == mode]
    for model, label, col, mk in [("naive", "naive", SERIES[1], "o"), ("pu_true_c", "PU at true c", SERIES[0], "s")]:
        bins = pd.DataFrame([b for r in sel for b in r["models"][model]["reliability"]])
        g = bins.groupby("bin")[["mean_p", "rate_y"]].mean()
        ax.plot(g["mean_p"], g["rate_y"], marker=mk, color=col, label=label, markeredgecolor="white", markersize=7)
    lim = [1e-5, 0.2]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Mean predicted P(y=1) in bin",
           title=f"{mode} dropout, q = {q}")
axes[0].set(ylabel="True connection rate in bin")
axes[0].legend(loc="upper left")
finish(fig, "B3_fafb_783_cx_benchmark-reliability_min10",
       "Both models are under-confident: too high in the low bins, too low in the top bin",
       "Dataset fafb 783; central-complex roster; τ = 10; q = 0.7; held-out sources, all deleted connections scored; "
       "bins of equal total sampling weight; mean over 5 seeds; dashed line = perfect calibration.")

"""Week 9 summary, tests and figures from reports/week9/runs_*.json (lock note addendum A2, A5, A8).

A condition is summarised only when its file holds exactly the 25 runs (seeds 10-14 x folds 0-4).
Writes reports/week9/summary.json, report/week9/numbers.tex and table_main.tex, and
figures/week9/fig1_reliability_<condition>.pdf (+ .png) and fig2_brier_by_condition.pdf.

Tests (A2, A5): the unit is the held-out source homologue group. For each group, D is its
weighted sum of pair Brier differences over all seeds, divided by its weight sum. Sign test of
P(D > 0 | D != 0) = 1/2, two-sided, ties dropped, alpha = 0.05; Holm over D1 and D2 in the
primary condition; D0 its own family; secondary conditions descriptive. Equal-weight mean of
group D with a 95% bootstrap interval over groups (B = 2,000), predictions held fixed.

Usage: python scripts/week9_report_numbers.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from scipy.stats import binomtest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS, FIG, TEX = ROOT / "reports" / "week9", ROOT / "figures" / "week9", ROOT / "report" / "week9"
PRIMARY = "synapse_q0.7"
CONDITIONS = {"q1": "no thinning ($q=1$)", "synapse_q0.7": "synapse $q=0.7$ (primary)",
              "connection_q0.7": "connection $q=0.7$", "synapse_q0.5": "synapse $q=0.5$"}
MODELS = {"baseline": "baseline", "naive": "naive", "naive_c_true": "naive/$c$", "pu": "PU"}
CONTRASTS = {"D0": ("naive", "naive_c_true"), "D1": ("naive", "pu"), "D2": ("naive_c_true", "pu")}
COLORS = {"naive": "#2a6fdb", "naive_c_true": "#d9822b", "pu": "#2e9e5b", "baseline": "#555555"}
B, ALPHA = 2000, 0.05
SEEDS, FOLDS = range(10, 15), range(5)


def complete(runs) -> bool:
    keys = [(r["seed"], r["fold"]) for r in runs]
    return len(keys) == len(set(keys)) and set(keys) == {(s, f) for s in SEEDS for f in FOLDS}


def per_seed_total(runs, get) -> float:
    """Held-out totals are disjoint across the folds of a seed: sum the folds, average the seeds."""
    return float(np.mean([sum(get(r) for r in runs if r["seed"] == s) for s in SEEDS]))


def group_d(runs, a, b) -> dict:
    """Per homologue group: (SSE_a - SSE_b) summed over runs / weight summed over runs."""
    num, den = {}, {}
    for r in runs:
        for g, v in r["by_source_group"].items():
            num[g] = num.get(g, 0.0) + v["sse"][a] - v["sse"][b]
            den[g] = den.get(g, 0.0) + v["weight"]
    return {g: num[g] / den[g] for g in num if den[g] > 0}


def sign_test(d: dict, seed: int = 0) -> dict:
    x = np.array(list(d.values()))
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    p = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else None          # all ties: no test
    boot = x[np.random.default_rng(seed).integers(0, len(x), size=(B, len(x)))].mean(1)
    return {"groups": len(x), "positive": pos, "negative": neg, "ties": len(x) - pos - neg, "p": None if p is None else float(p),
            "mean": float(x.mean()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}


def holm(ps: dict) -> dict:
    """Holm-adjusted p-values (step-down, monotone)."""
    order, m, out, run = sorted(ps, key=ps.get), len(ps), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def pooled_reliability(runs, model):
    """Bin-wise mean over runs of the 10 equal-weight bins: mean prediction, true rate and its SD."""
    bins = {}
    for r in runs:
        for row in r["models"][model]["reliability"]:
            bins.setdefault(row["bin"], []).append((row["mean_p"], row["rate_y"]))
    k = sorted(bins)
    p = np.array([np.mean([t[0] for t in bins[b]]) for b in k])
    y = np.array([np.mean([t[1] for t in bins[b]]) for b in k])
    sd = np.array([np.std([t[1] for t in bins[b]], ddof=1) if len(bins[b]) > 1 else 0.0 for b in k])
    return p, y, sd


def ms(x, scale=1.0, digits=3):
    x = np.asarray(x, float) * scale
    return f"{x.mean():.{digits}f} ({x.std(ddof=1):.{digits}f})" if len(x) > 1 else f"{x.mean():.{digits}f}"


def macro(name, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def write_numbers(summary):
    """Numeric macros for the report; prose is written in the .tex itself."""
    s, out = summary[PRIMARY], []
    out.append(macro("CPrimary", f"{s['realised_c_train'][0]:.3f}"))
    for m, tag in (("naive", "Naive"), ("naive_c_true", "NaiveC"), ("pu", "PU"), ("baseline", "Base")):
        out.append(macro(f"Brier{tag}", f"{s['models'][m]['brier_y'][0] * 1e3:.3f}"))
        out.append(macro(f"MeanP{tag}", f"{s['models'][m]['mean_p'][0]:.4f}"))
    out.append(macro("MeanY", f"{s['models']['naive']['mean_y'][0]:.4f}"))
    # Markov on the level identity: weighted share of pairs with p > 0.05 <= weighted mean p / 0.05
    # on f = g/c, per run; the smallest bound over runs, rounded down
    out.append(macro("PairsBelowFive", f"at least {math.floor(100 * s['share_f_le_0.05_bound']):.0f}\\%"))
    for k, v in s["tests"].items():
        if v is None:
            continue
        out.append(macro("TestD" + {"D0": "zero", "D1": "one", "D2": "two"}[k], f"{v['positive']}/{v['positive'] + v['negative']} groups positive, "
                                      f"$p={v.get('p_holm', v['p']):.3f}$, mean ${v['mean'] * 1e3:.3f}$ "
                                      f"$[{v['ci95'][0] * 1e3:.3f}, {v['ci95'][1] * 1e3:.3f}]\\times10^{{-3}}$"))
    # pair-weighted (pooled) difference over all held-out pairs, the quantity the table's Brier means compare
    for k, (a, b) in CONTRASTS.items():
        out.append(macro("Pooled" + {"D0": "Dzero", "D1": "Done", "D2": "Dtwo"}[k], f"{s['pooled_d'][k] * 1e3:.3f}"))
    if "q1" in summary:
        out.append(macro("TopPQone", f"{summary['q1']['top_bin']['naive']['mean_p']:.3f}"))
        out.append(macro("TopYQone", f"{summary['q1']['top_bin']['naive']['rate_y']:.3f}"))
    for m, tag in (("naive", "Naive"), ("naive_c_true", "NaiveC"), ("pu", "PU")):
        out.append(macro(f"Level{tag}", f"{s['level_ratio'][m][0]:.2f}"))
        out.append(macro(f"TopP{tag}", f"{s['top_bin'][m]['mean_p']:.3f}"))
    out.append(macro("TopY", f"{s['top_bin']['naive']['rate_y']:.3f}"))
    out.append(macro("LevelNaiveSD", f"{s['level_ratio']['naive'][1]:.2f}"))
    out.append(macro("AUCNaive", f"{s['models']['naive']['auc_y'][0]:.3f}"))
    out.append(macro("AUCPU", f"{s['models']['pu']['auc_y'][0]:.3f}"))
    times = "; ".join(f"{CONDITIONS[n]} {summary[n]['minutes']:.0f}~min" for n in CONDITIONS if n in summary)
    out.append(macro("RunTimes", f"mean over 25 runs on 8 CPU cores, {times}"))
    if "q1" in summary:
        q = summary["q1"]
        out.append(macro("LevelQone", f"{q['level_ratio']['naive'][0]:.2f}"))
        out.append(macro("BrierQone", f"{q['models']['naive']['brier_y'][0] * 1e3:.3f}"))
        out.append(macro("AUCQone", f"{q['models']['naive']['auc_y'][0]:.3f}"))
        out.append(macro("Recorded", f"{q['heldout_recorded']:,.0f}".replace(",", "{,}")))
        e = q["expected_missed"]
        for k, tag in (("0.63", "Mid"), ("0.53", "Lo"), ("0.78", "Hi")):
            out.append(macro(f"Missed{tag}", f"{round(e[k], -2):,.0f}".replace(",", "{,}")))
    a7 = RUNS / "edges_per_neuron.json"
    if a7.exists():
        d = json.loads(a7.read_text())
        same, scaled = d["settings"]["same_tau"], d["settings"]["scaled_tau"]
        out.append(macro("EdgesSentence", f"on {d['shared_types']} shared cell types, {d['fafb']['mean']:.1f} outgoing partners per neuron in FAFB "
                         f"and {same['mean']:.1f} in MaleCNS at $\\tau=10$ (ratio {same['ratio_mean']:.2f}), {scaled['mean']:.1f} at "
                         f"$\\tau={scaled['tau_malecns']}$ (ratio {scaled['ratio_mean']:.2f})"))
    (TEX / "numbers.tex").write_text("\n".join(out) + "\n")


def reliability_figure(runs, n, models, stem):
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    hi = 0.0
    for m in models:
        p, y, sd = pooled_reliability(runs, m)
        ax.errorbar(p, y, yerr=sd, marker="o", ms=3, lw=1.2, capsize=1.5, color=COLORS[m], label=MODELS[m].replace("$", ""))
        hi = max(hi, p.max(), (y + sd).max())
    bp = np.mean([r["models"]["baseline"]["mean_p"] for r in runs])
    by = np.mean([r["models"]["baseline"]["mean_y"] for r in runs])
    ax.plot([bp], [by], marker="s", color=COLORS["baseline"], ls="none", label="baseline")
    ax.plot([0, hi], [0, hi], color="#999999", lw=0.8, ls="--", label="perfect calibration")
    ax.set_xlabel("mean predicted P(connected), per bin")
    ax.set_ylabel("true connection rate, per bin")
    ax.set_title(CONDITIONS[n].replace("$", ""), fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def main():
    data = {n: json.loads((RUNS / f"runs_{n}.json").read_text()) for n in CONDITIONS if (RUNS / f"runs_{n}.json").exists()}
    for n in [n for n in data if not complete(data[n])]:
        print(f"{n}: {len(data[n])} runs, not the full 25; left out", file=sys.stderr)
        del data[n]
    if PRIMARY not in data:
        sys.exit("the primary condition is not complete yet")
    FIG.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    summary = {}
    for n, runs in data.items():
        s = {"runs": len(runs), "seeds": sorted({r["seed"] for r in runs}),
             "realised_c_train": [float(np.mean([r["realised_c_train"] for r in runs])), float(np.std([r["realised_c_train"] for r in runs], ddof=1))],
             "lambda": {m: {f"{lam:g}": sum(r["lambda"][m] == lam for r in runs) for lam in sorted({r["lambda"][m] for r in runs})}
                        for m in runs[0]["lambda"]},
             "pu_converged": all(r["pu_fit"]["converged"] for r in runs),
             "pu_inner_not_converged": sum(len(r["pu_fit"].get("inner_not_converged", [])) for r in runs),
             "pu_inner_checked": all("inner_fits" in r["pu_fit"] for r in runs),
             "train_mean_f": float(np.mean([r["train_mean_f"] for r in runs])),
             # Markov on f = g/c over the weighted training pairs: share(f > 0.05) <= mean(g) / (c * 0.05)
             "share_f_le_0.05_bound": float(max(0.0, min(1 - r["train_mean_f"] / (r["realised_c_train"] * 0.05) for r in runs))),
             "models": {m: {k: [float(np.mean([r["models"][m][k] for r in runs])), float(np.std([r["models"][m][k] for r in runs], ddof=1))]
                            for k in ("brier_y", "auc_y", "ap_y", "mean_p", "mean_y")} for m in MODELS},
             # level: per run, weighted mean prediction / weighted true rate on the held-out pairs
             "level_ratio": {m: [float(np.mean(v := [r["models"][m]["mean_p"] / r["models"][m]["mean_y"] for r in runs])),
                                 float(np.std(v, ddof=1))] for m in MODELS},
             # top reliability bin (the 10% of held-out weight with the highest predictions), pooled as in figure 1
             "top_bin": {m: dict(zip(("mean_p", "rate_y"), (float(a[-1]) for a in pooled_reliability(runs, m)[:2])))
                         for m in ("naive", "naive_c_true", "pu")},
             "minutes": float(np.mean([r["seconds"] for r in runs]) / 60),
             # pooled D: all held-out pairs of all runs, weighted as in the Brier score (groups not equal)
             "pooled_d": {k: float(sum(sum(g["sse"][a] - g["sse"][b] for g in r["by_source_group"].values()) for r in runs)
                                   / sum(sum(g["weight"] for g in r["by_source_group"].values()) for r in runs))
                          for k, (a, b) in CONTRASTS.items()},
             # at c = 1 PU is the naive model with naive's λ: D1 and D2 would test solver noise, so none
             "tests": {k: (None if k != "D0" and all(r["realised_c_train"] == 1 for r in runs) else sign_test(group_d(runs, a, b)))
                       for k, (a, b) in CONTRASTS.items()}}
        if n == PRIMARY:
            adj = holm({k: s["tests"][k]["p"] for k in ("D1", "D2")})
            for k, v in adj.items():
                s["tests"][k]["p_holm"] = v
        mirror = [r["c_mirror_train"] for r in runs if r["c_mirror_train"].get("pairs", 0) > 0]
        if mirror:
            iv = [m["interval"] for m in mirror if "interval" in m]
            s["c_mirror_train"] = {"share": float(np.mean([m["share"] for m in mirror])),
                                   "interval_mean": [float(np.mean([i[0] for i in iv])), float(np.mean([i[1] for i in iv]))] if iv else None,
                                   "full_graph_share": (float(np.mean([m["full_graph_share"] for m in mirror]))
                                                        if all("full_graph_share" in m for m in mirror) else None)}
        if n == "q1":
            s["expected_missed"] = {k: per_seed_total(runs, lambda r, k=k: r["real"]["expected_missed"][k])
                                    for k in runs[0]["real"]["expected_missed"] if all(k in r["real"]["expected_missed"] for r in runs)}
            s["heldout_recorded"] = per_seed_total(runs, lambda r: r["real"]["heldout_recorded"])
            s["heldout_unlabelled_weight"] = per_seed_total(runs, lambda r: r["real"]["heldout_unlabelled_weight"])
        summary[n] = s
        # figure 1 per condition: pooled reliability curves; A5's version (naive, naive/c) and one with PU (A8)
        for suffix, models in (("", ("naive", "naive_c_true", "pu")), ("_a5", ("naive", "naive_c_true"))):
            reliability_figure(runs, n, models, FIG / f"fig1_reliability_{n}{suffix}")
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=1))
    # figure 2: Brier per condition and model
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    names = [n for n in CONDITIONS if n in data]
    for j, m in enumerate(("naive", "naive_c_true", "pu")):
        x = np.arange(len(names)) + (j - 1) * 0.25
        vals = [[r["models"][m]["brier_y"] * 1e3 for r in data[n]] for n in names]
        ax.errorbar(x, [np.mean(v) for v in vals], yerr=[np.std(v, ddof=1) for v in vals], fmt="o", ms=3, capsize=2,
                    color=COLORS[m], label=MODELS[m].replace("$", ""))
    ax.set_xticks(np.arange(len(names)), [CONDITIONS[n].replace("$", "").replace(" (primary)", "*") for n in names], fontsize=6)
    ax.set_ylabel("Brier vs true labels (x1000)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig2_brier_by_condition.{ext}", dpi=200)
    plt.close(fig)
    # table
    rows = [r"\begin{tabular}{@{}llrrrr@{}}\toprule", r"condition & model & Brier & mean $p$ & AUC & AP\\\midrule"]
    for n in names:
        for m, lab in MODELS.items():
            runs = data[n]
            rows.append(f"{CONDITIONS[n] if m == 'baseline' else ''} & {lab} & {ms([r['models'][m]['brier_y'] for r in runs], 1e3)} & "
                        f"{ms([r['models'][m]['mean_p'] for r in runs], 1, 4)} & "
                        f"{ms([r['models'][m]['auc_y'] for r in runs], 1, 3) if m != 'baseline' else '0.5'} & "
                        f"{ms([r['models'][m]['ap_y'] for r in runs], 1, 3) if m != 'baseline' else '--'}\\\\")
        rows.append(r"\midrule" if n != names[-1] else r"\bottomrule")
    rows.append(r"\end{tabular}")
    (TEX / "table_main.tex").write_text("\n".join(rows) + "\n")
    write_numbers(summary)
    print(json.dumps({n: {"tests": s["tests"], "brier": {m: s["models"][m]["brier_y"] for m in MODELS},
                          "c": s["realised_c_train"], "lambda": s["lambda"]} for n, s in summary.items()}, indent=1))


if __name__ == "__main__":
    main()

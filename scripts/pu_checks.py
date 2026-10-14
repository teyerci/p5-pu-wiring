"""Appendix numbers for the NumPy PU module (src/pu.py): gradient check, agreement
with scikit-learn at c = 1, recovery on synthetic SCAR data, Elkan–Noto bias.
Writes reports/pu_checks.json; the same cases are asserted in tests/test_pu.py."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pu import PULogistic, elkan_noto_c, gradient_check   # noqa: E402
from tests.test_pu import scar_data                            # noqa: E402

out = {}
X, _, s, _, _ = scar_data(n=500, c=0.6, seed=1)
a = np.random.default_rng(2).uniform(0.5, 3.0, size=len(s))
out["gradient_check_max_rel_error"] = {str(c): gradient_check(X, s, a, c=c, lam=0.05) for c in (1.0, 0.7, 0.4)}

X, _, s, _, _ = scar_data(n=3000, seed=5)
a = np.random.default_rng(8).uniform(0.5, 3.0, size=len(s))
lam = 0.01
ours = PULogistic(c=1.0, lam=lam).fit(X, s, sample_weight=a)
ref = LogisticRegression(C=1 / (lam * a.sum()), tol=1e-12, max_iter=10000).fit(X, s, sample_weight=a)
out["sklearn_c1"] = {"max_abs_diff": float(max(np.abs(ours.coef_ - ref.coef_[0]).max(),
                                               abs(ours.intercept_ - ref.intercept_[0]))),
                     "newton_iterations": ours.n_iter_}

c = 0.5
X, y, s, w, b = scar_data(n=60000, c=c, seed=6)
pu = PULogistic(c=c, lam=1e-6).fit(X, s)
naive = PULogistic(c=1.0, lam=1e-6).fit(X, s)
out["recovery"] = {"c": c, "true_coef": w.tolist(), "true_intercept": b,
                   "pu_coef": pu.coef_.tolist(), "pu_intercept": pu.intercept_,
                   "naive_coef": naive.coef_.tolist(), "naive_intercept": naive.intercept_,
                   "mean_true_p": float(y.mean()), "mean_pu_p": float(pu.predict_proba(X).mean()),
                   "mean_naive_p": float(naive.predict_proba(X).mean())}

X, _, s, _, _ = scar_data(n=60000, c=0.6, seed=7)
half = len(s) // 2
g = PULogistic(c=1.0, lam=1e-6).fit(X[:half], s[:half])
out["elkan_noto"] = {"true_c": 0.6, **elkan_noto_c(g.predict_proba(X[half:][s[half:] == 1]), g.predict_proba(X[half:]))}

(ROOT / "reports" / "pu_checks.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

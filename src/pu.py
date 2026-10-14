"""PU-corrected logistic regression in NumPy (the course derivation, Project 5).

Setting (SCAR). A pair with true label y = 1 is recorded (s = 1) with constant
probability c, independent of x; s = 0 otherwise. Then P(s=1|x) = c·P(y=1|x).

Risk. With logistic loss and z = wᵀx, the negative-class risk over the
unlabelled data is written as a mixture, which gives the convex unbiased PU risk
(du Plessis, Niu and Sugiyama 2015) using only recorded positives P and all
examples X:

    R(w) = -π · E_P[z] + E_X[log(1 + e^z)] + (λ/2) wᵀΛw,    π = P(s=1) / c,

where π is the class prior P(y=1). E_X and E_P are weighted by the sampling
weights a (inverse inclusion), P(s=1) is the weighted share of recorded
positives, and Λ = diag(1, …, 1, 0) leaves the intercept unpenalised.

    ∇R = -π E_P[x] + E_X[σ(z) x] + λΛw,     ∇²R = E_X[σ(z)(1-σ(z)) x xᵀ] + λΛ.

The Hessian is positive semi-definite, so R is convex; it is bounded below when
0 < π < 1 (and λ > 0), which is the guard on c: c must exceed P(s=1). At c = 1,
π = P(s=1) and R is the weighted logistic loss, so the fit equals scikit-learn's
LogisticRegression(C = 1 / (λ Σa), sample_weight = a) up to solver tolerance.

The model returns f(x) = σ(wᵀx), an estimate of P(y=1|x); c·f(x) estimates the
observable P(s=1|x).
"""
from __future__ import annotations

import warnings

import numpy as np


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))           # stable for large |z|


def _design(X):
    X = np.asarray(X, dtype=float)
    return np.hstack([X, np.ones((X.shape[0], 1))])  # intercept last


def class_prior(s, a, c) -> float:
    """π = P(s=1) / c with weighted P(s=1); raises unless 0 < π < 1 (c must exceed P(s=1))."""
    p_s = float(np.sum(a * s) / np.sum(a))
    pi = p_s / c
    if not 0 < pi < 1:
        raise ValueError(f"infeasible c = {c}: π = P(s=1)/c = {pi:.4g} must lie in (0, 1); "
                         f"c must exceed P(s=1) = {p_s:.4g}")
    return pi


def pu_risk(w, Xd, s, a, pi, lam, with_hessian=False):
    """Value, gradient (and Hessian) of the PU risk; Xd includes the intercept column."""
    z = Xd @ w
    wx, wp = a / a.sum(), (a * s) / (a * s).sum()
    pen = np.ones_like(w); pen[-1] = 0.0
    log1pexp = np.logaddexp(0.0, z)
    value = -pi * wp @ z + wx @ log1pexp + 0.5 * lam * np.sum(pen * w * w)
    p = sigmoid(z)
    grad = -pi * (wp @ Xd) + (wx * p) @ Xd + lam * pen * w
    if not with_hessian:
        return value, grad
    hess = (Xd * (wx * p * (1 - p))[:, None]).T @ Xd + lam * np.diag(pen)
    return value, grad, hess


class PULogistic:
    """Logistic model for P(y=1|x) fitted with the PU risk by damped Newton.

    c = 1 gives ordinary (weighted, L2) logistic regression on the recorded labels:
    the naive model. `lam` is the L2 strength on the coefficients (not the intercept).
    """

    def __init__(self, c: float = 1.0, lam: float = 1e-3, tol: float = 1e-10, max_iter: int = 100,
                 tol_decrement: float = 1e-15):
        if not 0 < c <= 1:
            raise ValueError("c must lie in (0, 1]")
        if not lam > 0:
            raise ValueError("lam must be > 0: with λ = 0 the risk can be unbounded below for c < 1")
        self.c, self.lam, self.tol, self.max_iter, self.tol_decrement = c, lam, tol, max_iter, tol_decrement

    def fit(self, X, s, sample_weight=None):
        Xd = _design(X)
        s = np.asarray(s, dtype=float)
        a = np.ones(len(s)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if s.sum() == 0:
            raise ValueError("no recorded positives")
        self.pi_ = class_prior(s, a, self.c)
        w = np.zeros(Xd.shape[1])
        value, grad, hess = pu_risk(w, Xd, s, a, self.pi_, self.lam, with_hessian=True)
        self.n_iter_ = 0
        converged = False
        for it in range(self.max_iter):
            step = np.linalg.solve(hess + 1e-12 * np.eye(len(w)), -grad)
            decrease = grad @ step                         # minus the Newton decrement (≤ 0)
            # stop on a small gradient, or when the predicted decrease is at round-off level:
            # near the optimum the gradient cannot fall below ~1e-10 in float64 on large designs
            if np.max(np.abs(grad)) < self.tol or -decrease < self.tol_decrement:
                converged = True
                break
            t = 1.0
            while True:                                    # Armijo backtracking (damping)
                new = w + t * step
                v_new = pu_risk(new, Xd, s, a, self.pi_, self.lam)[0]
                if v_new <= value + 1e-4 * t * decrease or t < 1e-10:
                    break
                t *= 0.5
            w = new
            value, grad, hess = pu_risk(w, Xd, s, a, self.pi_, self.lam, with_hessian=True)
            self.n_iter_ = it + 1
        self.converged_ = converged or bool(np.max(np.abs(grad)) < self.tol)
        if not self.converged_:
            warnings.warn(f"PULogistic did not converge in {self.n_iter_} iterations "
                          f"(max |gradient| {np.max(np.abs(grad)):.2e})", RuntimeWarning, stacklevel=2)
        self.coef_, self.intercept_, self.risk_ = w[:-1], float(w[-1]), float(value)
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        """f(x): estimated P(y=1|x), the probability of a true connection."""
        return sigmoid(self.decision_function(X))

    def predict_recorded(self, X):
        """c · f(x): estimated P(s=1|x), the probability that a connection is recorded."""
        return self.c * self.predict_proba(X)


def elkan_noto_c(g_heldout_positives, g_all=None) -> dict:
    """Elkan and Noto (2008) estimators of c from a naive model g(x) ≈ P(s=1|x).

    e1 = mean of g over held-out recorded positives. Under SCAR and for a calibrated
    g = c·f, e1 = c · E_P[f(x)] ≤ c, with equality only if f(x) = 1 on every positive
    (positives separable); otherwise e1 is a lower bound (in expectation).
    e3 = max of g over all examples: one extreme score decides it, so it is high
    variance and, with a misspecified or overconfident g, can exceed c.
    """
    g = np.asarray(g_heldout_positives, dtype=float)
    out = {"e1": float(g.mean()), "n_heldout": int(len(g))}
    if g_all is not None:
        out["e3"] = float(np.max(g_all))
    return out


def gradient_check(X, s, a=None, c=1.0, lam=1e-2, eps=1e-6, seed=0) -> float:
    """Largest relative difference between the analytic gradient and central finite
    differences, at a random point (for the appendix)."""
    rng = np.random.default_rng(seed)
    Xd = _design(X)
    s = np.asarray(s, dtype=float)
    a = np.ones(len(s)) if a is None else np.asarray(a, dtype=float)
    pi = class_prior(s, a, c)
    w = rng.normal(scale=0.5, size=Xd.shape[1])
    _, g = pu_risk(w, Xd, s, a, pi, lam)
    num = np.empty_like(w)
    for k in range(len(w)):
        e = np.zeros_like(w); e[k] = eps
        num[k] = (pu_risk(w + e, Xd, s, a, pi, lam)[0] - pu_risk(w - e, Xd, s, a, pi, lam)[0]) / (2 * eps)
    return float(np.max(np.abs(num - g) / np.maximum(1e-8, np.abs(num) + np.abs(g))))

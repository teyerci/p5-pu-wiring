"""Checks the proposal requires of the NumPy PU module: finite-difference gradient,
exact agreement with scikit-learn at c = 1, recovery of known parameters on synthetic
SCAR data, and a guard against infeasible c."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.pu import PULogistic, class_prior, elkan_noto_c, gradient_check, pu_risk, _design


def scar_data(n=40000, c=0.6, seed=0, w=(1.5, -2.0, 0.8), b=-1.0):
    """y ~ Bernoulli(σ(w·x + b)); recorded s = y · Bernoulli(c) (SCAR)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(w)))
    p = 1 / (1 + np.exp(-(X @ np.asarray(w) + b)))
    y = rng.random(n) < p
    s = y & (rng.random(n) < c)
    return X, y.astype(float), s.astype(float), np.asarray(w), b


@pytest.mark.parametrize("c", [1.0, 0.7, 0.4])
def test_gradient_matches_finite_differences(c):
    X, _, s, _, _ = scar_data(n=500, c=0.6, seed=1)
    a = np.random.default_rng(2).uniform(0.5, 3.0, size=len(s))
    assert gradient_check(X, s, a, c=c, lam=0.05) < 1e-6


def test_hessian_matches_finite_differences_of_the_gradient():
    X, _, s, _, _ = scar_data(n=300, seed=3)
    Xd, a = _design(X), np.ones(len(s))
    pi = class_prior(s, a, 0.5)
    w = np.random.default_rng(4).normal(size=Xd.shape[1])
    _, _, H = pu_risk(w, Xd, s, a, pi, 0.1, with_hessian=True)
    num = np.column_stack([(pu_risk(w + e, Xd, s, a, pi, 0.1)[1] - pu_risk(w - e, Xd, s, a, pi, 0.1)[1]) / 2e-6
                           for e in np.eye(len(w)) * 1e-6])
    assert np.allclose(num, H, atol=1e-6)


def test_c_equal_1_matches_scikit_learn_weighted_logistic_regression():
    X, _, s, _, _ = scar_data(n=3000, seed=5)
    a = np.random.default_rng(8).uniform(0.5, 3.0, size=len(s))   # non-constant among positives too
    lam = 0.01
    ours = PULogistic(c=1.0, lam=lam).fit(X, s, sample_weight=a)
    ref = LogisticRegression(C=1 / (lam * a.sum()), tol=1e-12, max_iter=10000).fit(X, s, sample_weight=a)
    assert ours.converged_
    assert np.allclose(ours.coef_, ref.coef_[0], atol=1e-6)
    assert abs(ours.intercept_ - ref.intercept_[0]) < 1e-6


def test_recovers_known_parameters_under_scar_and_the_naive_model_does_not():
    X, y, s, w, b = scar_data(n=60000, c=0.5, seed=6)
    pu = PULogistic(c=0.5, lam=1e-6).fit(X, s)
    assert np.allclose(pu.coef_, w, atol=0.15) and abs(pu.intercept_ - b) < 0.15
    naive = PULogistic(c=1.0, lam=1e-6).fit(X, s)
    # the naive model estimates P(s=1|x) = c·P(y=1|x), so it underestimates connection probability
    assert naive.predict_proba(X).mean() < 0.75 * y.mean() < pu.predict_proba(X).mean() * 1.1


def test_infeasible_c_is_refused():
    s = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)  # P(s=1) = 0.3
    with pytest.raises(ValueError, match="infeasible"):
        class_prior(s, np.ones(10), c=0.25)
    with pytest.raises(ValueError, match="infeasible"):
        PULogistic(c=0.3).fit(np.zeros((10, 1)), s)
    assert class_prior(s, np.ones(10), c=0.6) == pytest.approx(0.5)


def test_elkan_noto_e1_is_a_lower_bound_when_positives_are_not_separable():
    X, _, s, _, _ = scar_data(n=60000, c=0.6, seed=7)
    half = len(s) // 2
    naive = PULogistic(c=1.0, lam=1e-6).fit(X[:half], s[:half])
    held = X[half:][s[half:] == 1]
    est = elkan_noto_c(naive.predict_proba(held), naive.predict_proba(X[half:]))
    assert est["e1"] < 0.6                               # c·E_P[f] < c: overlap between the classes
    assert est["e3"] > 0.6                               # the max score overshoots the true c


def test_non_convergence_warns_and_lambda_zero_is_refused():
    X, _, s, _, _ = scar_data(n=2000, seed=9)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        m = PULogistic(c=0.6, lam=1e-3, max_iter=1).fit(X, s)
    assert not m.converged_
    with pytest.raises(ValueError, match="lam"):
        PULogistic(c=0.6, lam=0.0)


def test_newton_decrement_stops_a_fit_whose_gradient_cannot_reach_the_tolerance():
    import warnings
    X, _, s, _, _ = scar_data(n=3000, seed=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # no non-convergence warning
        m = PULogistic(c=0.7, lam=1e-3, tol=0.0).fit(X, s)               # tol 0: only the decrement can stop it
    assert m.converged_ and m.n_iter_ < 30

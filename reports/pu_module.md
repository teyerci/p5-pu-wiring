# NumPy PU module — derivation summary and checks (Project 5)

`src/pu.py`; checks in `tests/test_pu.py`; numbers from `scripts/pu_checks.py` (also in `reports/pu_checks.json`).

## Derivation (as implemented)

1. **SCAR.** A true connection (y = 1) is recorded (s = 1) with constant probability c, independent of x. Unrecorded pairs (s = 0) mix true negatives with missed positives.
   - Biologically, this assumes detection does not depend on synapse size, location or neuron type.
   - The discrepancy count (step 15) already shows that it is imperfect: connections near τ flip far more often than strong ones.
2. **Observable vs true probability.** P(s=1|x) = P(s=1|y=1,x)·P(y=1|x) = c·P(y=1|x), since a recorded pair is always a true connection. A classifier trained on s therefore estimates c·P(y=1|x). Weights, regularisation or thresholds change the fit, not this target.
3. **Estimating c (Elkan and Noto).** Fit g(x) ≈ P(s=1|x), then average it over held-out recorded positives: e1 = E_P[g] = c·E_P[f].
   - For a calibrated g, e1 equals c only if f(x) = 1 on every positive, that is, if positives are separable. Otherwise e1 is a lower bound (in expectation).
   - The maximum score e3 = max g(x) is decided by one extreme example. It is high-variance, and with a misspecified or overconfident g it can exceed c.
4. **Corrected risk.** Write the risk on the negatives, E_N[ℓ(z,−1)], as the unlabelled mixture: (1−π)E_N[ℓ(z,−1)] = E_X[ℓ(z,−1)] − π·E_P[ℓ(z,−1)]. Every term is then computable from P and X. With logistic loss, ℓ(z,+1) − ℓ(z,−1) = −z, and the risk becomes

   R(w) = −π·E_P[z] + E_X[log(1+e^z)] + (λ/2)wᵀΛw,  π = P(s=1)/c,

   with sampling weights in E_X and E_P and an unpenalised intercept.
   - **Gradient:** −π·E_P[x] + E_X[σ(z)x] + λΛw.
   - **Hessian:** E_X[σ(1−σ)xxᵀ] + λΛ. It is positive semi-definite, so R is convex.
   - R is bounded below when 0 < π < 1 (λ > 0), so c must exceed P(s=1). The code refuses any other c.
5. **c = 1 reduction.** π = P(s=1), and R is the weighted logistic loss on the recorded labels. It matches scikit-learn's `LogisticRegression(C = 1/(λ·Σa), sample_weight = a)`.

The fit is damped Newton: a Newton step with Armijo backtracking. It stops when every gradient component is below 10⁻¹⁰, or when the Newton decrement is below 10⁻¹⁵. On large designs the gradient stalls near 10⁻¹⁰ through round-off, while the risk no longer changes; the pilot's c = 0.7 fit showed this. A fit that stops for neither reason raises a warning.

## Checks

| Check | Result |
|---|---|
| Finite-difference gradient (central, ε = 10⁻⁶), c = 1, 0.7, 0.4, random weights | max relative error **2.4·10⁻⁹, 2.3·10⁻⁸, 7.7·10⁻¹⁰** |
| Hessian vs finite differences of the gradient | agree to 10⁻⁶ |
| c = 1 vs scikit-learn (weighted, L2, C = 1/(λΣa)) | max coefficient difference **5.0·10⁻⁹** (weights non-constant on both classes); 6 Newton steps |
| Recovery, synthetic SCAR, c = 0.5, n = 60,000 | true w = (1.5, −2.0, 0.8), b = −1.0; PU (1.57, −2.10, 0.83), −0.99; **naive (0.75, −1.00, 0.39), −1.88** |
| Mean P(y=1\|x) | true 0.375; PU 0.378; **naive 0.189** |
| Elkan–Noto, true c = 0.6 | **e1 = 0.39** (a lower bound: the classes overlap); **e3 = 0.98** (the max overshoots) |
| Infeasible c (c ≤ P(s=1)) | refused with an error |

**What this shows for the project:**
- **The naive model** halves the connection probability at c = 0.5. It also roughly halves every coefficient, so the distortion is not only a constant: feature effects are attenuated. This is the bias the brief says no regularisation or threshold can fix.
- **The PU model** recovers the true parameters when c is known.
- **c itself** is not identifiable from e1 or e3 alone. That is why the proposal reports performance at known c, the estimators' bias on the dropout benchmark, and a sensitivity range on real data (H1).

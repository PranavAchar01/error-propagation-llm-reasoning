"""Inference: logistic slopes, paired bootstrap, McNemar, correction, power.

No network, no model. Everything here is a deterministic function of the raw
records plus a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

# Small ridge penalty on the slope. Depth-conditioned accuracy can separate
# perfectly in a small sample (everything right at depth 1, everything wrong at
# depth 5), which sends the unpenalised MLE to infinity. The penalty keeps beta
# finite and comparable across conditions; it shrinks toward zero, so it is
# conservative with respect to the hypothesis that beta differs from zero.
RIDGE = 1e-3


def fit_logit(x: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> tuple[float, float]:
    """Ridge-penalised logistic fit of y ~ alpha + beta*x, via IRLS.

    Returns (alpha, beta). Hand-rolled because the paired bootstrap refits this
    tens of thousands of times and a general-purpose GLM's overhead dominates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(np.unique(x)) < 2:
        return float("nan"), float("nan")

    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    penalty = np.diag([0.0, ridge])  # never penalise the intercept

    for _ in range(60):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        grad = X.T @ (y - p) - penalty @ beta
        hess = X.T @ (X * w[:, None]) + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(beta[0]), float(beta[1])


@dataclass
class BootResult:
    point: float
    lo: float
    hi: float
    p_value: float
    n: int
    n_resamples: int

    def fmt(self) -> str:
        return f"{self.point:+.3f} [95% CI {self.lo:+.3f}, {self.hi:+.3f}], p={self.p_value:.4f}, n={self.n}"


def paired_bootstrap_delta_beta(
    depth: np.ndarray,
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootResult:
    """Bootstrap the difference in depth slopes between two conditions.

    Resampling is clustered by ITEM: the same item is drawn for both conditions
    together, which is what makes the comparison paired. Resampling the two
    conditions independently would discard the pairing and inflate the variance.

    `correct_a` is the intervention, `correct_b` the baseline, so a positive
    delta means the intervention's accuracy decays more slowly with depth.
    """
    depth = np.asarray(depth, dtype=float)
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    n = len(depth)
    if n == 0:
        return BootResult(float("nan"), float("nan"), float("nan"), float("nan"), 0, n_resamples)

    point = fit_logit(depth, a)[1] - fit_logit(depth, b)[1]

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples)
    deltas[:] = np.nan
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        d = depth[idx]
        if len(np.unique(d)) < 2:
            continue
        deltas[i] = fit_logit(d, a[idx])[1] - fit_logit(d, b[idx])[1]

    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < 100:
        return BootResult(point, float("nan"), float("nan"), float("nan"), n, n_resamples)

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # Two-sided bootstrap p: how often the resampled effect crosses zero,
    # doubled. Clipped to a floor of 1/B — a bootstrap cannot resolve below its
    # own resolution, and reporting p=0 would overstate what B resamples show.
    frac = float(np.mean(deltas <= 0)) if point > 0 else float(np.mean(deltas >= 0))
    p = min(1.0, max(2 * frac, 1.0 / len(deltas)))
    return BootResult(point, float(lo), float(hi), p, n, len(deltas))


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar on paired binary outcomes. Returns (b, c, p).

    b = items the intervention got right and the baseline wrong; c = the
    reverse. Only discordant pairs carry information, which is exactly why the
    test is paired.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    if b + c == 0:
        return b, c, 1.0
    p = float(sps.binomtest(b, b + c, 0.5).pvalue)
    return b, c, p


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Correct at the extremes, unlike normal approx."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down. Uniformly more powerful than plain Bonferroni."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    prev = 0.0
    for i, (name, p) in enumerate(items):
        thresh = alpha / (m - i)
        adj = max(prev, min(1.0, p * (m - i)))
        prev = adj
        out[name] = {"p_raw": p, "p_adjusted": adj, "threshold": thresh, "reject": adj <= alpha}
    return out


def mde_delta_beta(
    depth: np.ndarray,
    baseline_correct: np.ndarray,
    target_power: float = 0.80,
    alpha: float = 0.05,
    n_sims: int = 400,
    seed: int = 0,
) -> float:
    """Minimum detectable delta-beta at the realised n and depth distribution.

    This is what separates "no effect" from "underpowered to see one". We
    simulate an intervention whose slope differs by a candidate delta, and find
    the smallest delta detected at least `target_power` of the time. If the
    returned MDE is larger than the pre-registered predicted effect, the study
    could not have detected its own hypothesis, and the report says so.
    """
    depth = np.asarray(depth, dtype=float)
    y = np.asarray(baseline_correct, dtype=float)
    if len(depth) == 0 or len(np.unique(depth)) < 2:
        return float("nan")

    alpha0, beta0 = fit_logit(depth, y)
    if not np.isfinite(beta0):
        return float("nan")

    rng = np.random.default_rng(seed)
    zcrit = sps.norm.ppf(1 - alpha / 2)

    for delta in np.arange(0.05, 3.01, 0.05):
        hits = 0
        for _ in range(n_sims):
            p_base = 1 / (1 + np.exp(-(alpha0 + beta0 * depth)))
            p_int = 1 / (1 + np.exp(-(alpha0 + (beta0 + delta) * depth)))
            yb = rng.binomial(1, np.clip(p_base, 0, 1)).astype(float)
            yi = rng.binomial(1, np.clip(p_int, 0, 1)).astype(float)
            d_hat = fit_logit(depth, yi)[1] - fit_logit(depth, yb)[1]
            # Cheap normal-approx SE from a small inner bootstrap.
            inner = np.empty(24)
            for j in range(24):
                idx = rng.integers(0, len(depth), len(depth))
                if len(np.unique(depth[idx])) < 2:
                    inner[j] = np.nan
                    continue
                inner[j] = fit_logit(depth[idx], yi[idx])[1] - fit_logit(depth[idx], yb[idx])[1]
            se = np.nanstd(inner)
            if np.isfinite(d_hat) and se > 0 and abs(d_hat / se) > zcrit:
                hits += 1
        if hits / n_sims >= target_power:
            return float(delta)
    return float("inf")

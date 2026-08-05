"""Statistics tests.

The analysis is where a study most easily fools itself, so each estimator is
checked against a case whose answer is known independently of the code.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from epr.stats import (
    fit_logit,
    fit_logit_se,
    holm_bonferroni,
    mcnemar_exact,
    mde_delta_beta,
    paired_bootstrap_delta_beta,
    wilson_ci,
)


def test_logit_recovers_planted_coefficients():
    rng = np.random.default_rng(0)
    x = rng.integers(1, 6, 8000).astype(float)
    alpha_true, beta_true = 2.0, -0.6
    p = 1 / (1 + np.exp(-(alpha_true + beta_true * x)))
    y = rng.binomial(1, p).astype(float)
    alpha, beta = fit_logit(x, y)
    assert alpha == pytest.approx(alpha_true, abs=0.15)
    assert beta == pytest.approx(beta_true, abs=0.08)


def test_logit_is_finite_under_perfect_separation():
    """Small samples can separate perfectly; beta must stay finite and comparable."""
    x = np.array([1.0, 1, 2, 2, 5, 5])
    y = np.array([1.0, 1, 1, 1, 0, 0])
    _, beta = fit_logit(x, y)
    assert np.isfinite(beta)
    assert beta < 0


def test_logit_returns_nan_when_depth_does_not_vary():
    """A slope on a constant is undefined and must not silently come back as 0."""
    _, beta = fit_logit(np.ones(50), np.zeros(50))
    assert np.isnan(beta)


def test_bootstrap_detects_a_planted_slope_difference():
    rng = np.random.default_rng(1)
    depth = np.repeat([1.0, 2, 3, 4, 5], 200)
    p_base = 1 / (1 + np.exp(-(3.0 - 0.9 * depth)))
    p_int = 1 / (1 + np.exp(-(3.0 - 0.3 * depth)))  # flatter -> delta = +0.6
    a = rng.binomial(1, p_int).astype(float)
    b = rng.binomial(1, p_base).astype(float)
    r = paired_bootstrap_delta_beta(depth, a, b, n_resamples=1500, seed=2)
    assert r.point == pytest.approx(0.6, abs=0.25)
    assert r.lo > 0, "CI should exclude zero for a real effect"
    assert r.p_value < 0.05


def test_bootstrap_does_not_manufacture_an_effect_under_the_null():
    rng = np.random.default_rng(3)
    depth = np.repeat([1.0, 2, 3, 4, 5], 120)
    p = 1 / (1 + np.exp(-(2.0 - 0.5 * depth)))
    a = rng.binomial(1, p).astype(float)
    b = rng.binomial(1, p).astype(float)
    r = paired_bootstrap_delta_beta(depth, a, b, n_resamples=1500, seed=4)
    assert r.p_value > 0.05
    assert r.lo < 0 < r.hi


def test_bootstrap_p_value_never_reports_zero():
    """A bootstrap cannot resolve below 1/B; reporting p=0 would overstate it."""
    rng = np.random.default_rng(5)
    depth = np.repeat([1.0, 2, 3, 4, 5], 200)
    a = (depth < 6).astype(float)
    b = rng.binomial(1, 1 / (1 + np.exp(-(3.0 - 1.5 * depth)))).astype(float)
    r = paired_bootstrap_delta_beta(depth, a, b, n_resamples=500, seed=6)
    assert r.p_value >= 1.0 / 500


def test_mcnemar_matches_the_exact_binomial():
    a = np.array([1] * 8 + [0] * 1 + [1] * 5 + [0] * 5, dtype=bool)
    b = np.array([0] * 8 + [1] * 1 + [1] * 5 + [0] * 5, dtype=bool)
    disc_b, disc_c, p = mcnemar_exact(a, b)
    assert (disc_b, disc_c) == (8, 1)
    assert p == pytest.approx(sps.binomtest(8, 9, 0.5).pvalue)


def test_mcnemar_ignores_concordant_pairs():
    """Only disagreements carry information — that is what makes it paired."""
    a = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
    b = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
    assert mcnemar_exact(a, b) == (1, 1, 1.0)


def test_wilson_ci_is_sane_at_the_boundaries():
    lo, hi = wilson_ci(0, 30)
    assert lo == 0.0 and 0 < hi < 0.2
    lo, hi = wilson_ci(30, 30)
    assert hi == 1.0 and 0.8 < lo < 1.0
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi


def test_holm_bonferroni_matches_a_hand_computed_example():
    out = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03}, alpha=0.05)
    assert out["a"]["p_adjusted"] == pytest.approx(0.03)
    assert out["c"]["p_adjusted"] == pytest.approx(0.06)
    assert out["b"]["p_adjusted"] == pytest.approx(0.06)
    assert out["a"]["reject"] is True
    assert out["b"]["reject"] is False and out["c"]["reject"] is False


def test_holm_is_monotone_and_never_below_raw():
    out = holm_bonferroni({"x": 0.001, "y": 0.2, "z": 0.9})
    assert all(v["p_adjusted"] >= v["p_raw"] for v in out.values())


def test_mde_shrinks_with_sample_size():
    """More data must make smaller effects detectable.

    `inf` is a legitimate answer, not a failure: it means no effect in the
    searched range is detectable at this n. Reporting a finite MDE there would
    overstate what the study could have seen.
    """
    rng = np.random.default_rng(7)
    small = np.repeat([1.0, 2, 3, 4, 5], 12)
    large = np.repeat([1.0, 2, 3, 4, 5], 400)
    ys = rng.binomial(1, 1 / (1 + np.exp(-(2.0 - 0.5 * small)))).astype(float)
    yl = rng.binomial(1, 1 / (1 + np.exp(-(2.0 - 0.5 * large)))).astype(float)
    mde_small = mde_delta_beta(small, ys, n_sims=60, seed=8)
    mde_large = mde_delta_beta(large, yl, n_sims=60, seed=8)
    assert np.isfinite(mde_large), "a large sample must detect *something*"
    assert mde_large < mde_small


def test_analytic_slope_se_matches_the_bootstrap_it_replaced():
    """The power analysis swapped a bootstrap SE for the Fisher-information SE.

    That was a 25x speedup, so it has to be the same quantity — otherwise the
    MDE, and the underpowered-vs-null call that rests on it, silently shifts.
    """
    rng = np.random.default_rng(11)
    x = np.repeat([1.0, 2, 3, 4, 5], 200)
    y = rng.binomial(1, 1 / (1 + np.exp(-(2.0 - 0.5 * x)))).astype(float)

    _, _, se_analytic = fit_logit_se(x, y)

    boots = np.empty(300)
    for i in range(300):
        idx = rng.integers(0, len(x), len(x))
        boots[i] = fit_logit(x[idx], y[idx])[1]
    se_boot = float(np.std(boots))

    assert se_analytic == pytest.approx(se_boot, rel=0.15), (
        f"analytic SE {se_analytic:.4f} vs bootstrap {se_boot:.4f}"
    )

"""End-to-end analysis test on synthetic records with a *known* planted effect.

This proves two things that matter for the paper:
  1. `make reproduce` runs with no network and no API key.
  2. The pipeline recovers an effect that was put there on purpose, and does not
     invent one when there is none.

The records here are synthetic and live only in a temp directory. Nothing in
results/ is ever produced by this test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from epr import metrics as metrics_mod
from epr.metrics import load_records, paired_vectors, summarise
from epr.stats import paired_bootstrap_delta_beta

ROOT = Path(__file__).resolve().parents[1]


def _record(
    uid, dataset, condition, seed, depth, correct, *, n_steps=3, first_error=None, propagated=0
):
    attempt = {
        "text": "synthetic",
        "answer": "True" if correct else "False",
        "correct": bool(correct),
        "input_tokens": 100,
        "output_tokens": 50,
        "error": "",
        "n_steps": n_steps,
        "parse_failed": False,
        "n_malformed": 0,
        "report": {
            "n_steps": n_steps,
            "first_error_index": first_error,
            "n_invalid": 0 if first_error is None else 1,
            "n_propagated": propagated,
            "all_valid": first_error is None,
            "verdicts": ["ok"] * n_steps,
            "grounded": [True] * n_steps,
        },
    }
    return {
        "uid": uid,
        "dataset": dataset,
        "condition": condition,
        "seed": seed,
        "model": "synthetic-model-v0",
        "depth": depth,
        "gold_answer": "True",
        "supports_step_metrics": True,
        "first": attempt,
        "revised": None,
        "meta": {},
    }


@pytest.fixture
def synthetic_phase(tmp_path, monkeypatch):
    """Two conditions over identical items; `struct_verify` has a flatter slope."""
    rng = np.random.default_rng(42)
    raw = tmp_path / "raw" / "synthetic" / "prontoqa"
    raw.mkdir(parents=True)

    rows = {"cot_fs": [], "struct_verify": []}
    for depth in (1, 2, 3, 4, 5):
        for i in range(120):
            uid = f"item_d{depth}_{i}"
            p_base = 1 / (1 + np.exp(-(3.0 - 0.9 * depth)))
            p_int = 1 / (1 + np.exp(-(3.0 - 0.3 * depth)))  # planted delta = +0.6
            cb = int(rng.random() < p_base)
            ci = int(rng.random() < p_int)
            rows["cot_fs"].append(
                _record(
                    uid,
                    "prontoqa",
                    "cot_fs",
                    1,
                    depth,
                    cb,
                    first_error=None if cb else 2,
                    propagated=0 if cb else 1,
                )
            )
            rows["struct_verify"].append(
                _record(
                    uid,
                    "prontoqa",
                    "struct_verify",
                    1,
                    depth,
                    ci,
                    first_error=None if ci else 3,
                    propagated=0,
                )
            )
    for cond, recs in rows.items():
        (raw / f"{cond}_seed1.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")

    monkeypatch.setattr(metrics_mod, "RESULTS", tmp_path)
    return tmp_path


def test_pipeline_recovers_a_planted_slope_difference(synthetic_phase):
    records = load_records("synthetic")
    assert len(records) == 1200

    depth, ca, cb, uids = paired_vectors(records, "struct_verify", "cot_fs", "prontoqa")
    assert len(uids) == 600, "conditions must pair on identical items"

    r = paired_bootstrap_delta_beta(depth, ca, cb, n_resamples=1200, seed=0)
    assert r.point == pytest.approx(0.6, abs=0.3), f"planted +0.6, recovered {r.point:+.3f}"
    assert r.lo > 0 and r.p_value < 0.05


def test_summaries_carry_correct_denominators(synthetic_phase):
    summaries = summarise(load_records("synthetic"))
    s = summaries[("prontoqa", "struct_verify")]
    assert s.n == 600
    assert s.n_correct == sum(s.accuracy_by_depth()[d][0] for d in s.accuracy_by_depth())
    # recovery denominator is items WITH an error, never all items
    assert s.n_with_error <= s.n
    if s.recovery_rate is not None:
        assert 0.0 <= s.recovery_rate <= 1.0
    lo, hi = s.accuracy_ci
    assert lo < s.accuracy < hi


def test_beta_is_more_negative_for_the_steeper_condition(synthetic_phase):
    summaries = summarise(load_records("synthetic"))
    beta_base = summaries[("prontoqa", "cot_fs")].beta_depth
    beta_int = summaries[("prontoqa", "struct_verify")].beta_depth
    assert beta_base < beta_int < 0, "the intervention should decay more slowly"


def test_metrics_are_blank_not_zero_when_unmeasurable(tmp_path, monkeypatch):
    """FOLIO-shaped records have no gold chain; step metrics must be None."""
    raw = tmp_path / "raw" / "nostep" / "folio"
    raw.mkdir(parents=True)
    rec = _record("f1", "folio", "cot_fs", 1, None, True)
    rec["supports_step_metrics"] = False
    rec["first"]["report"] = None
    (raw / "cot_fs_seed1.jsonl").write_text(json.dumps(rec) + "\n")
    monkeypatch.setattr(metrics_mod, "RESULTS", tmp_path)

    s = summarise(load_records("nostep"))[("folio", "cot_fs")]
    assert s.n == 1
    assert s.recovery_rate is None
    assert s.mean_first_error_position is None
    assert s.verifier_rejection_rate is None
    assert s.beta_depth is None


def test_analyze_script_runs_offline_and_writes_tables(synthetic_phase, monkeypatch):
    """`make reproduce` must work with no API key present."""
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "PYTHONPATH": str(ROOT / "src"),
        "EPR_RESULTS_ROOT": str(synthetic_phase),
        "HOME": str(synthetic_phase),
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze.py"),
            "--phase",
            "synthetic",
            "--bootstrap",
            "300",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        check=False,
    )
    assert "ANTHROPIC_API_KEY" not in env
    assert proc.returncode == 0, f"analyze.py failed:\n{proc.stdout}\n{proc.stderr}"
    assert "Per-condition summary" in proc.stdout
    assert "synthetic-model-v0" in proc.stdout

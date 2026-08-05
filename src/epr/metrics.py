"""Raw records -> the pre-registered metrics. Offline, deterministic.

Every metric carries its own denominator. A rate whose denominator is zero is
reported as None and rendered blank, never as 0.0 — the difference between "we
measured zero" and "there was nothing to measure" is the whole point of the
post-error recovery metric.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .stats import fit_logit, wilson_ci

# Overridable so the analysis pipeline can be exercised against a temp tree in
# tests without ever touching the real results/.
RESULTS = Path(
    os.environ.get("EPR_RESULTS_ROOT") or Path(__file__).resolve().parents[2] / "results"
)


def load_records(phase: str, dataset: str | None = None) -> list[dict]:
    """Read every raw record for a phase. This is the only input to analysis."""
    root = RESULTS / "raw" / phase
    if not root.exists():
        return []
    out: list[dict] = []
    for path in sorted(root.rglob("*.jsonl")):
        if dataset and path.parent.name != dataset:
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def final_attempt(rec: dict) -> dict:
    """The attempt a condition is scored on: post-revision where one exists."""
    return rec["revised"] if rec.get("revised") else rec["first"]


@dataclass
class ConditionSummary:
    dataset: str
    condition: str
    model: str
    seeds: list[int] = field(default_factory=list)

    n: int = 0
    n_correct: int = 0
    n_unparsed_answer: int = 0
    n_api_error: int = 0

    # step-level; denominators differ from n and are tracked explicitly
    n_with_steps: int = 0
    n_parse_failed: int = 0
    n_with_error: int = 0
    n_recovered: int = 0
    first_error_positions: list[int] = field(default_factory=list)
    n_steps_total: int = 0
    n_steps_invalid: int = 0
    n_steps_propagated: int = 0

    depths: list[int] = field(default_factory=list)
    corrects: list[int] = field(default_factory=list)

    # matched-compute view: first attempt only
    n_first_correct: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.n_correct / self.n if self.n else None

    @property
    def accuracy_ci(self) -> tuple[float, float]:
        return wilson_ci(self.n_correct, self.n)

    @property
    def first_attempt_accuracy(self) -> float | None:
        return self.n_first_correct / self.n if self.n else None

    @property
    def parse_failure_rate(self) -> float | None:
        return self.n_parse_failed / self.n_with_steps if self.n_with_steps else None

    @property
    def recovery_rate(self) -> float | None:
        """Correct final answer despite a broken chain. High = chain is decorative."""
        return self.n_recovered / self.n_with_error if self.n_with_error else None

    @property
    def mean_first_error_position(self) -> float | None:
        return float(np.mean(self.first_error_positions)) if self.first_error_positions else None

    @property
    def verifier_rejection_rate(self) -> float | None:
        return self.n_steps_invalid / self.n_steps_total if self.n_steps_total else None

    @property
    def propagation_rate(self) -> float | None:
        return self.n_steps_propagated / self.n_steps_total if self.n_steps_total else None

    @property
    def beta_depth(self) -> float | None:
        """The pre-registered propagation number: slope of accuracy on depth."""
        if len(set(self.depths)) < 2:
            return None
        _, beta = fit_logit(np.array(self.depths), np.array(self.corrects))
        return None if not np.isfinite(beta) else float(beta)

    def accuracy_by_depth(self) -> dict[int, tuple[int, int]]:
        out: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for d, c in zip(self.depths, self.corrects, strict=True):
            out[d][1] += 1
            out[d][0] += c
        return {d: (v[0], v[1]) for d, v in sorted(out.items())}


def summarise(records: list[dict]) -> dict[tuple[str, str], ConditionSummary]:
    """Aggregate raw records into one summary per (dataset, condition)."""
    out: dict[tuple[str, str], ConditionSummary] = {}

    for rec in records:
        key = (rec["dataset"], rec["condition"])
        s = out.get(key)
        if s is None:
            s = out[key] = ConditionSummary(rec["dataset"], rec["condition"], rec["model"])
        if rec["seed"] not in s.seeds:
            s.seeds.append(rec["seed"])

        first = rec["first"]
        att = final_attempt(rec)

        if att.get("error"):
            s.n_api_error += 1
            continue

        s.n += 1
        correct = bool(att.get("correct"))
        s.n_correct += correct
        s.n_first_correct += bool(first.get("correct"))
        if att.get("answer") is None:
            s.n_unparsed_answer += 1

        if rec.get("depth") is not None:
            s.depths.append(int(rec["depth"]))
            s.corrects.append(int(correct))

        if not rec.get("supports_step_metrics"):
            continue

        s.n_with_steps += 1
        if att.get("parse_failed"):
            s.n_parse_failed += 1
            continue

        report = att.get("report")
        if not report:
            continue

        s.n_steps_total += report["n_steps"]
        s.n_steps_invalid += report["n_invalid"]
        s.n_steps_propagated += report["n_propagated"]

        fei = report.get("first_error_index")
        if fei is not None:
            s.n_with_error += 1
            s.first_error_positions.append(int(fei))
            if correct:
                s.n_recovered += 1

    return out


def paired_vectors(
    records: list[dict], cond_a: str, cond_b: str, dataset: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Align two conditions on the items they share.

    Returns (depth, correct_a, correct_b, uids) over the intersection only.
    Comparing conditions on different item sets would break the pairing that
    both the bootstrap and McNemar depend on.
    """
    by_cond: dict[str, dict[str, dict]] = {cond_a: {}, cond_b: {}}
    for rec in records:
        if rec["dataset"] != dataset or rec["condition"] not in by_cond:
            continue
        if final_attempt(rec).get("error"):
            continue
        by_cond[rec["condition"]][f"{rec['uid']}#{rec['seed']}"] = rec

    shared = sorted(set(by_cond[cond_a]) & set(by_cond[cond_b]))
    depth, ca, cb, uids = [], [], [], []
    for k in shared:
        ra, rb = by_cond[cond_a][k], by_cond[cond_b][k]
        if ra.get("depth") is None:
            continue
        depth.append(int(ra["depth"]))
        ca.append(int(bool(final_attempt(ra).get("correct"))))
        cb.append(int(bool(final_attempt(rb).get("correct"))))
        uids.append(k)
    return np.array(depth), np.array(ca), np.array(cb), uids


def contamination_flag(summary: ConditionSummary, threshold: float = 0.95) -> str | None:
    """Flag a baseline so strong it suggests memorisation rather than reasoning."""
    acc = summary.accuracy
    if acc is not None and acc >= threshold and summary.condition.startswith("direct"):
        return (
            f"{summary.dataset}/{summary.condition}: direct-answer accuracy "
            f"{acc:.1%} (n={summary.n}) — implausibly high without reasoning; "
            "possible contamination or leakage."
        )
    return None

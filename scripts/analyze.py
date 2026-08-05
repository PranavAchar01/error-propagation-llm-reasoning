#!/usr/bin/env python3
"""Regenerate every table from results/raw/. No network, no model, no API key.

    python scripts/analyze.py --phase pilot

Writes results/tables/*.md and *.csv. Numbers that cannot be computed from the
raw records are rendered as "--", never imputed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epr.metrics import (
    contamination_flag,
    load_records,
    paired_vectors,
    summarise,
)
from epr.prompts import CONDITIONS
from epr.stats import (
    holm_bonferroni,
    mcnemar_exact,
    mde_delta_beta,
    paired_bootstrap_delta_beta,
)

# Honour the same override the metrics loader uses, so a test (or a scratch
# analysis) can never write into the real results/tables/. Synthetic output
# sitting next to real output is an integrity hazard, not a tidiness issue.
TABLES = Path(os.environ.get("EPR_RESULTS_ROOT") or (ROOT / "results")) / "tables"

# The three confirmatory comparisons carrying the Holm-Bonferroni family.
# Pre-registered in HYPOTHESIS.md §4; do not add to this list post hoc.
CONFIRMATORY = (
    ("C1", "prontoqa", "struct_verify", "cot_fs", "delta_beta"),
    ("C2", "proofwriter", "struct_verify", "cot_fs", "delta_beta"),
    ("C3", "prontoqa", "struct_verify", "cot_fs", "accuracy"),
)


def pct(x: float | None, n: int | None = None) -> str:
    if x is None:
        return "--"
    return f"{100 * x:.1f}%" + (f" (n={n})" if n is not None else "")


def num(x: float | None, fmt: str = "{:+.3f}") -> str:
    return "--" if x is None else fmt.format(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="pilot")
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = load_records(args.phase)
    if not records:
        print(
            f"No raw records for phase '{args.phase}'.\n"
            f"Expected files under results/raw/{args.phase}/<dataset>/*.jsonl\n"
            "Run `make pilot` first (needs ANTHROPIC_API_KEY)."
        )
        return 1

    TABLES.mkdir(parents=True, exist_ok=True)
    summaries = summarise(records)
    datasets = sorted({d for d, _ in summaries})
    model = next(iter(summaries.values())).model
    out: list[str] = [
        f"# Results — phase `{args.phase}`",
        "",
        (
            f"Model: `{model}`  ·  raw records: {len(records):,}  ·  "
            f"seeds: {sorted({r['seed'] for r in records})}"
        ),
        "",
        "Every percentage carries its denominator. `--` means the quantity is not",
        "measurable for that dataset (FOLIO and BBH ship no gold proof chain), not zero.",
        "",
    ]

    # ---------------------------------------------------------------- main table
    out += ["## Per-condition summary", ""]
    header = (
        "| dataset | condition | n | accuracy [95% CI] | first-attempt acc | "
        "beta_depth | first-error pos | post-error recovery | verifier reject | parse fail |"
    )
    out += [header, "|" + "---|" * 10]
    rows_csv = []
    for ds in datasets:
        for cond in CONDITIONS:
            s = summaries.get((ds, cond))
            if s is None:
                continue
            lo, hi = s.accuracy_ci
            out.append(
                f"| {ds} | `{cond}` | {s.n} | {pct(s.accuracy)} [{pct(lo)}, {pct(hi)}] | "
                f"{pct(s.first_attempt_accuracy)} | {num(s.beta_depth)} | "
                f"{num(s.mean_first_error_position, '{:.2f}')} | "
                f"{pct(s.recovery_rate, s.n_with_error)} | "
                f"{pct(s.verifier_rejection_rate, s.n_steps_total)} | "
                f"{pct(s.parse_failure_rate, s.n_with_steps)} |"
            )
            rows_csv.append(
                {
                    "dataset": ds,
                    "condition": cond,
                    "n": s.n,
                    "accuracy": s.accuracy,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "first_attempt_accuracy": s.first_attempt_accuracy,
                    "beta_depth": s.beta_depth,
                    "mean_first_error_position": s.mean_first_error_position,
                    "n_with_error": s.n_with_error,
                    "recovery_rate": s.recovery_rate,
                    "n_steps_total": s.n_steps_total,
                    "verifier_rejection_rate": s.verifier_rejection_rate,
                    "propagation_rate": s.propagation_rate,
                    "n_with_steps": s.n_with_steps,
                    "parse_failure_rate": s.parse_failure_rate,
                    "n_api_error": s.n_api_error,
                    "n_unparsed_answer": s.n_unparsed_answer,
                    "model": s.model,
                }
            )
    out.append("")

    with (TABLES / f"summary_{args.phase}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)

    # ------------------------------------------------------- accuracy by depth
    out += ["## Accuracy by ground-truth proof depth", ""]
    for ds in datasets:
        depth_conds = [
            (c, summaries[(ds, c)])
            for c in CONDITIONS
            if (ds, c) in summaries and summaries[(ds, c)].depths
        ]
        if not depth_conds:
            continue
        all_depths = sorted({d for _, s in depth_conds for d in s.depths})
        out += [
            f"### {ds}",
            "",
            "| condition | " + " | ".join(f"d={d}" for d in all_depths) + " | beta |",
        ]
        out.append("|" + "---|" * (len(all_depths) + 2))
        for cond, s in depth_conds:
            byd = s.accuracy_by_depth()
            cells = []
            for d in all_depths:
                if d in byd:
                    k, n = byd[d]
                    cells.append(f"{100 * k / n:.0f}% ({k}/{n})")
                else:
                    cells.append("--")
            out.append(f"| `{cond}` | " + " | ".join(cells) + f" | {num(s.beta_depth)} |")
        out.append("")

    # -------------------------------------------------- confirmatory tests
    out += ["## Confirmatory tests (pre-registered)", ""]
    pvals: dict[str, float] = {}
    details: dict[str, str] = {}
    for tag, ds, a, b, kind in CONFIRMATORY:
        depth, ca, cb, uids = paired_vectors(records, a, b, ds)
        if len(uids) == 0:
            details[tag] = f"**{tag}** ({ds}, `{a}` vs `{b}`, {kind}): no paired items — not run."
            continue
        if kind == "delta_beta":
            r = paired_bootstrap_delta_beta(depth, ca, cb, args.bootstrap, args.seed)
            pvals[tag] = r.p_value
            details[tag] = (
                f"**{tag}** ({ds}, delta_beta `{a}` - `{b}`): {r.fmt()}, "
                f"{r.n_resamples:,} resamples"
            )
        else:
            bb, cc, p = mcnemar_exact(ca, cb)
            pvals[tag] = p
            delta = (ca.mean() - cb.mean()) if len(ca) else float("nan")
            details[tag] = (
                f"**{tag}** ({ds}, accuracy `{a}` - `{b}`): {100 * delta:+.1f} points, "
                f"exact McNemar b={bb} c={cc}, p={p:.4f}, n={len(ca)}"
            )

    corrected = holm_bonferroni(pvals) if pvals else {}
    for tag, _, _, _, _ in CONFIRMATORY:
        line = details.get(tag, f"**{tag}**: not run.")
        if tag in corrected:
            c = corrected[tag]
            verdict = "REJECT H0" if c["reject"] else "fail to reject H0"
            line += f"  ·  Holm-adjusted p={c['p_adjusted']:.4f} -> **{verdict}**"
        out += [f"- {line}", ""]

    # ----------------------------------------------------------- power / MDE
    out += ["## Power", ""]
    for ds in ("prontoqa", "proofwriter"):
        depth, ca, cb, uids = paired_vectors(records, "struct_verify", "cot_fs", ds)
        if len(uids) < 20:
            out.append(f"- {ds}: too few paired items ({len(uids)}) to estimate power.")
            continue
        mde = mde_delta_beta(depth, cb, seed=args.seed)
        verdict = (
            "the study is **underpowered for its own primary hypothesis** "
            "(MDE exceeds the pre-registered prediction of +0.15)"
            if mde > 0.15
            else "the study is powered to detect the pre-registered effect"
        )
        out += [
            (
                f"- **{ds}**: n={len(uids)} paired items. Minimum detectable delta_beta "
                f"at 80% power = **{mde:+.2f}** logits/depth. Therefore {verdict}."
            ),
            "",
        ]

    # ------------------------------------------------------- contamination
    flags = [f for s in summaries.values() if (f := contamination_flag(s))]
    out += ["## Contamination checks", ""]
    out += [f"- ⚠️ {f}" for f in flags] if flags else ["- No direct-answer baseline exceeded 95%."]
    out.append("")

    # ---------------------------------------------------------- run failures
    errs = sum(s.n_api_error for s in summaries.values())
    unp = sum(s.n_unparsed_answer for s in summaries.values())
    out += [
        "## Run health",
        "",
        f"- API calls that failed after retries and were excluded: **{errs}**",
        f"- Responses with no parseable ANSWER line (scored incorrect): **{unp}**",
        "",
    ]

    path = TABLES / f"results_{args.phase}.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[analyze] wrote {path} and summary_{args.phase}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

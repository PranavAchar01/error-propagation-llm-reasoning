#!/usr/bin/env python3
"""Run an experiment phase. Prints a cost projection and refuses to overspend.

    python scripts/run.py --phase pilot --n 50  --seeds 1
    python scripts/run.py --phase full  --n 300 --seeds 1 2 3

`--dry-run` builds every prompt and prints the projection without calling the
API, which is also the fastest way to check the harness end to end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from epr.datasets import Item, load_bbh, load_folio, load_prontoqa, load_proofwriter
from epr.model import (
    DEFAULT_MODEL,
    PRICING,
    Client,
    MissingCredentials,
    estimate_cost,
    provider_for,
)
from epr.prompts import CONDITIONS, FEWSHOT, VERIFIED, build_prompt
from epr.runner import BudgetExceeded, output_path, run_condition

ROOT = Path(__file__).resolve().parents[1]

N_EXEMPLARS = 4
DATASETS = ("prontoqa", "proofwriter", "folio", "bbh")

# Mean output tokens per call, by condition family. Used only for the pre-run
# projection; actuals are recorded per call and reported afterwards.
EXPECTED_OUT = {"direct": 12, "cot": 260, "struct": 380}


def _family(condition: str) -> str:
    if condition.startswith("direct"):
        return "direct"
    if condition in ("struct", "struct_verify"):
        return "struct"
    return "cot"


def load_dataset(name: str, seed: int, n: int) -> tuple[list[Item], list[Item], str]:
    """Return (eval_items, exemplars, load_summary). The two never overlap."""
    if name == "prontoqa":
        hops = [1, 2, 3, 4, 5]
        per = max(1, n // len(hops))
        items, rep = load_prontoqa(hops, seed, per + 2)
    elif name == "proofwriter":
        depths = (0, 1, 2, 3, 4, 5)
        per = max(1, n // len(depths))
        items, rep = load_proofwriter(seed, per + 2, depths)
    elif name == "folio":
        items, rep = load_folio(seed, n + N_EXEMPLARS)
    elif name == "bbh":
        items, rep = load_bbh(seed, max(1, (n + N_EXEMPLARS) // 3))
    else:
        raise SystemExit(f"unknown dataset {name}")

    # Hold out exemplars first, then evaluate on the rest. Where the dataset has
    # depth labels the exemplars are spread across depths so the few-shot block
    # is not implicitly a hint that every problem is shallow; where it has none
    # (FOLIO, BBH) they are simply taken from the tail.
    by_depth: dict[object, list[Item]] = {}
    for it in items:
        by_depth.setdefault(it.depth, []).append(it)

    exemplars: list[Item] = []
    if set(by_depth) == {None}:
        exemplars = items[-N_EXEMPLARS:]
    else:
        for d in sorted(by_depth, key=lambda x: (x is None, x)):
            if len(exemplars) < N_EXEMPLARS and by_depth[d]:
                cand = by_depth[d][-1]
                # Only items with a usable gold chain can serve as worked examples.
                if cand.supports_step_metrics:
                    exemplars.append(cand)
        # Depth-balanced picks can come up short if some depths lack gold proofs.
        for it in reversed(items):
            if len(exemplars) >= N_EXEMPLARS:
                break
            if it not in exemplars and it.supports_step_metrics:
                exemplars.append(it)

    held = {id(e) for e in exemplars}
    eval_items = [i for i in items if id(i) not in held][:n]
    return eval_items, exemplars, rep.summary()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["pilot", "full"])
    ap.add_argument("--n", type=int, default=50, help="eval items per dataset per seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--datasets", nargs="+", default=["prontoqa", "proofwriter"], choices=DATASETS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-cost", type=float, default=25.0, help="USD ceiling for this phase")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    for c in args.conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c}; choose from {CONDITIONS}")

    # ---- build every prompt up front so the projection is real, not a guess
    plan: list[tuple[str, str, int, list[Item], list[Item]]] = []
    prompts: list[tuple[str, str]] = []
    expected_out = 0
    print(f"\n=== phase={args.phase} model={args.model} provider={provider_for(args.model)} ===")
    for ds in args.datasets:
        for seed in args.seeds:
            items, exemplars, summary = load_dataset(ds, seed, args.n)
            print(f"  seed {seed} | {summary} | eval n={len(items)} exemplars={len(exemplars)}")
            for cond in args.conditions:
                plan.append((ds, cond, seed, items, exemplars))
                ex = exemplars if cond in FEWSHOT else []
                for it in items:
                    p = build_prompt(it, cond, ex)
                    prompts.append((p.system, p.user))
                    out = EXPECTED_OUT[_family(cond)]
                    # verified conditions may spend a second call on revision
                    expected_out += out * (2 if cond in VERIFIED else 1)

    tin, _, _ = estimate_cost(prompts, args.model, 0)
    # revision calls resend the premise block plus the prior attempt
    tin_total = int(tin * 1.6) if any(c in VERIFIED for c in args.conditions) else tin
    cin, cout = PRICING.get(args.model, (0.0, 0.0))
    projected = tin_total / 1e6 * cin + expected_out / 1e6 * cout

    print(
        f"\n  planned calls    : {len(prompts):,} (+ up to {sum(1 for d, c, s, i, e in plan if c in VERIFIED) * args.n:,} revisions)"
    )
    print(f"  projected tokens : {tin_total:,} in / {expected_out:,} out")
    print(f"  PROJECTED COST   : ${projected:.2f}  (ceiling ${args.max_cost:.2f})")

    if args.dry_run:
        print("\n  dry run: no API calls made.")
        return 0

    if projected > args.max_cost and not args.yes:
        print(
            f"\n  REFUSING TO RUN: ${projected:.2f} exceeds the ${args.max_cost:.2f} ceiling.\n"
            "  Re-run with a smaller --n, fewer --seeds/--conditions, or an explicit "
            "--max-cost if this spend is intended."
        )
        return 2

    try:
        client = Client(model=args.model)
    except MissingCredentials as e:
        print(f"\n  {e}")
        return 3

    # A projection is an estimate; only real usage enforces a real ceiling.
    # Checked before every item, so a hard budget is actually hard.
    def over_budget() -> bool:
        return client.usage.cost(args.model) >= args.max_cost

    total = len(prompts)
    stopped = False
    with tqdm(total=total, desc=args.phase, unit="item") as bar:
        for ds, cond, seed, items, exemplars in plan:
            ex = exemplars if cond in FEWSHOT else []
            try:
                run_condition(
                    client,
                    items,
                    cond,
                    ex,
                    seed,
                    args.phase,
                    progress=bar,
                    stop_check=over_budget,
                )
            except BudgetExceeded as e:
                print(f"\n\n  BUDGET CEILING HIT: {e}")
                print("  Completed records are on disk; re-run to resume where this stopped.")
                stopped = True
                break

    print(f"\n  actual usage: {client.usage.summary(args.model)}")
    if stopped:
        print("  INCOMPLETE RUN — analysis will report reduced n for the affected conditions.")
    print(
        f"  raw output  : {output_path(args.datasets[0], args.conditions[0], args.seeds[0], args.phase).parent.parent}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

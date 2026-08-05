"""Benchmark loaders producing one uniform `Item` type.

Every quirk documented in HYPOTHESIS.md §6 is handled here, explicitly and
loudly. Nothing is silently coerced: an item that cannot be loaded soundly is
dropped and counted, and the counts are reported alongside the results.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .logic import Atom, Theory
from .parsers import (
    ParseError,
    UnsupportedTheory,
    Vocab,
    build_prontoqa_theory,
    build_proofwriter_theory,
    parse_prontoqa_sentence,
    prontoqa_gold_steps,
    proofwriter_gold_steps,
)
from .verifier import Step

DATA = Path(__file__).resolve().parents[2] / "data"

# The three-way label space, normalised across every dataset. FOLIO calls the
# third class `Unknown` in train and `Uncertain` in validation; ProofWriter uses
# Python booleans plus the string "Unknown".
TRUE, FALSE, UNKNOWN = "True", "False", "Unknown"


def normalise_label(raw) -> str:
    if isinstance(raw, bool):
        return TRUE if raw else FALSE
    s = str(raw).strip().lower()
    if s in ("true", "entailment", "yes"):
        return TRUE
    if s in ("false", "contradiction", "no"):
        return FALSE
    if s in ("unknown", "uncertain", "neutral", "undetermined"):
        return UNKNOWN
    raise ValueError(f"unrecognised label: {raw!r}")


@dataclass
class Item:
    """One benchmark question, uniform across datasets."""

    uid: str
    dataset: str
    question: str
    gold_answer: str
    theory: Theory | None = None
    gold_steps: list[Step] = field(default_factory=list)
    depth: int | None = None
    answer_space: tuple[str, ...] = (TRUE, FALSE)
    vocab: Vocab | None = None
    meta: dict = field(default_factory=dict)

    @property
    def premise_block(self) -> str:
        return self.theory.premise_block() if self.theory else self.meta.get("premises_text", "")

    @property
    def supports_step_metrics(self) -> bool:
        """Whether first-error position / recovery can be scored for this item.

        False for FOLIO and BBH, which ship no gold proof. Those items appear in
        accuracy tables and are *blank* in step-level tables — never estimated.
        """
        return self.theory is not None and bool(self.gold_steps)

    def parse_proposition(self, text: str) -> Atom:
        """Dataset-specific reader for a model's stated conclusion."""
        if self.dataset == "prontoqa":
            parsed = parse_prontoqa_sentence(text, self.vocab or Vocab())
            if not isinstance(parsed, Atom):
                raise ParseError("conclusion is a rule, not a proposition")
            return parsed
        if self.dataset == "proofwriter":
            return _parse_pw_english(text)
        raise ParseError(f"{self.dataset} has no proposition parser")


@dataclass
class LoadReport:
    """What was loaded and, importantly, what was not."""

    dataset: str
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        drop = ", ".join(f"{k}={v}" for k, v in sorted(self.dropped.items())) or "none"
        return f"{self.dataset}: kept n={self.kept}, dropped: {drop}"


# --------------------------------------------------------------------------
# ProofWriter English <-> Atom
# --------------------------------------------------------------------------

_PW_RELATIONS = ("is", "eats", "sees", "chases", "likes", "needs", "visits")


def _parse_pw_english(text: str) -> Atom:
    """ "The cow is not big." -> Atom(cow, is, big, False).

    ProofWriter's surface form is as rigid as ProntoQA's, so this stays a
    grammar, not a heuristic.
    """
    t = text.strip().rstrip(".").strip()
    t = t.replace("The ", "").replace("the ", "")
    for rel in _PW_RELATIONS:
        marker = f" {rel} "
        if marker in t:
            subj, _, rest = t.partition(marker)
            pol = True
            rest = rest.strip()
            for neg in ("not ", "does not ", "doesn't "):
                if rest.startswith(neg):
                    pol, rest = False, rest[len(neg) :].strip()
                    break
            if not subj.strip() or not rest:
                break
            return Atom(subj.strip(), rel, rest.split()[0].strip(), pol)
    raise ParseError(f"unrecognised ProofWriter proposition: {text!r}")


# --------------------------------------------------------------------------
# ProntoQA
# --------------------------------------------------------------------------


def load_prontoqa(hops: list[int], seed: int, per_hop: int) -> tuple[list[Item], LoadReport]:
    """Load generated ProntoQA. Files come from `make data`, never the network."""
    rep = LoadReport("prontoqa")
    items: list[Item] = []
    for h in hops:
        path = DATA / "prontoqa_gen" / f"{h}hop_seed{seed}.json"
        if not path.exists():
            rep.dropped[f"missing_file_hop{h}"] += 1
            continue
        for key, ex in json.loads(path.read_text()).items():
            if len([i for i in items if i.depth == h]) >= per_hop:
                break
            te = ex["test_example"]
            try:
                theory, vocab = build_prontoqa_theory(te["question"])
                gold = prontoqa_gold_steps(te["chain_of_thought"], theory, vocab)
                answer = normalise_label(te["answer"])
            except (ParseError, ValueError) as e:
                rep.dropped[type(e).__name__] += 1
                continue
            items.append(
                Item(
                    uid=f"prontoqa_h{h}_s{seed}_{key}",
                    dataset="prontoqa",
                    question=te["query"],
                    gold_answer=answer,
                    theory=theory,
                    gold_steps=gold,
                    depth=h,
                    answer_space=(TRUE, FALSE),
                    vocab=vocab,
                    meta={"gold_chain": te["chain_of_thought"]},
                )
            )
            rep.kept += 1
    return items, rep


# --------------------------------------------------------------------------
# ProofWriter
# --------------------------------------------------------------------------

PW_PATH = DATA / "pw_extract/proofwriter-dataset-V2020.12.3/OWA/depth-5/meta-test.jsonl"


def load_proofwriter(
    seed: int, per_depth: int, depths=(0, 1, 2, 3, 4, 5)
) -> tuple[list[Item], LoadReport]:
    """Load ProofWriter OWA depth-5, balanced across QDep.

    Only True/False items carry a gold proof under the open-world assumption;
    `Unknown` items have none by construction. Unknown items are kept for
    accuracy (dropping them would bias the label distribution) but contribute no
    step-level metrics.
    """
    rep = LoadReport("proofwriter")
    if not PW_PATH.exists():
        rep.dropped["missing_file"] += 1
        return [], rep

    rng = random.Random(seed)
    buckets: dict[int, list[Item]] = {d: [] for d in depths}
    lines = PW_PATH.read_text().splitlines()
    rng.shuffle(lines)

    for line in lines:
        if all(len(v) >= per_depth for v in buckets.values()):
            break
        record = json.loads(line)
        try:
            theory = build_proofwriter_theory(record)
        except UnsupportedTheory:
            rep.dropped["negation_as_failure"] += 1
            continue
        except ParseError as e:
            rep.dropped[type(e).__name__] += 1
            continue

        for qid, q in record["questions"].items():
            d = q.get("QDep")
            if d not in buckets or len(buckets[d]) >= per_depth:
                continue
            try:
                gold = proofwriter_gold_steps(q, theory)
                answer = normalise_label(q["answer"])
            except (ParseError, ValueError) as e:
                rep.dropped[type(e).__name__] += 1
                continue
            buckets[d].append(
                Item(
                    uid=f"proofwriter_{record['id']}_{qid}",
                    dataset="proofwriter",
                    question=q["question"],
                    gold_answer=answer,
                    theory=theory,
                    gold_steps=gold,
                    depth=d,
                    answer_space=(TRUE, FALSE, UNKNOWN),
                    meta={"qdep": d},
                )
            )
            rep.kept += 1

    items = [i for d in sorted(buckets) for i in buckets[d]]
    for d, v in buckets.items():
        if len(v) < per_depth:
            rep.dropped[f"short_depth{d}"] += per_depth - len(v)
    return items, rep


# --------------------------------------------------------------------------
# FOLIO
# --------------------------------------------------------------------------

FOLIO_VAL = DATA / "folio_src/data/v0.0/folio-validation.jsonl"


def load_folio(seed: int, n: int) -> tuple[list[Item], LoadReport]:
    """FOLIO validation. No gold proof chain, therefore no step-level metrics.

    Premises are numbered `fact1..factN` so the prompt format matches the other
    datasets exactly, but there is no Theory: FOLIO's full first-order logic
    (quantifier alternation, XOR, nested implication) is outside the fragment
    this verifier soundly decides. Claiming otherwise would be the single
    easiest way to fabricate a result here.
    """
    rep = LoadReport("folio")
    if not FOLIO_VAL.exists():
        rep.dropped["missing_file"] += 1
        return [], rep

    rows = [json.loads(line) for line in FOLIO_VAL.read_text().splitlines()]
    random.Random(seed).shuffle(rows)
    items: list[Item] = []
    for i, r in enumerate(rows):
        if len(items) >= n:
            break
        try:
            answer = normalise_label(r["label"])
        except ValueError as e:
            rep.dropped[str(e)[:40]] += 1
            continue
        premises = "\n".join(f"fact{j + 1}: {p.strip()}" for j, p in enumerate(r["premises"]))
        items.append(
            Item(
                uid=f"folio_val_{i}",
                dataset="folio",
                question=f"True, false, or unknown: {r['conclusion'].strip()}",
                gold_answer=answer,
                theory=None,
                gold_steps=[],
                depth=None,
                answer_space=(TRUE, FALSE, UNKNOWN),
                meta={"premises_text": premises, "n_premises": len(r["premises"])},
            )
        )
        rep.kept += 1
    return items, rep


# --------------------------------------------------------------------------
# BIG-Bench Hard: logical_deduction (out-of-distribution check)
# --------------------------------------------------------------------------

BBH_DIR = DATA / "bbh"
_BBH_TASKS = {
    "logical_deduction_three_objects": 3,
    "logical_deduction_five_objects": 5,
    "logical_deduction_seven_objects": 7,
}


def load_bbh(seed: int, n_per_task: int) -> tuple[list[Item], LoadReport]:
    """BBH logical_deduction. Multiple choice, answer-accuracy only.

    Object count (3/5/7) is recorded as a *proxy* for difficulty, not as proof
    depth — these items have no proofs, so they cannot enter the slope analysis.
    """
    rep = LoadReport("bbh")
    items: list[Item] = []
    for task, n_objects in _BBH_TASKS.items():
        path = BBH_DIR / f"{task}.json"
        if not path.exists():
            rep.dropped[f"missing_{task}"] += 1
            continue
        rows = json.loads(path.read_text())["examples"]
        random.Random(seed).shuffle(rows)
        for i, r in enumerate(rows[:n_per_task]):
            items.append(
                Item(
                    uid=f"bbh_{task}_{i}",
                    dataset="bbh",
                    question=r["input"],
                    gold_answer=r["target"].strip(),
                    theory=None,
                    gold_steps=[],
                    depth=None,
                    answer_space=(),
                    meta={"task": task, "n_objects": n_objects},
                )
            )
            rep.kept += 1
    return items, rep

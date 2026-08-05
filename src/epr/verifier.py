"""The non-model verifier.

A step is a typed triple: (premise_ids, rule_id, derived_proposition). The
verifier resolves the cited ids against the premise set, checks that the cited
rule actually fires on those premises, and checks that the stated conclusion is
what the rule yields. A step citing premises that do not entail it is a
*detectable* error, not a stylistic one.

Two notions of correctness are tracked separately, and the distinction is what
makes error propagation measurable at all:

  local validity — is the inference sound *treating the cited premises as given*?
  groundedness   — are those premises themselves traceable back to the original
                   theory through a chain of locally-valid steps?

A step that is locally valid but not grounded is a propagated error: the model
reasoned correctly from something it had already gotten wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

from .logic import Atom, Theory, fire

# Above this many cited premises we stop searching subsets and report an arity
# error. Keeps a model from brute-forcing a pass by citing the whole theory.
_MAX_OVERCITE = 8


class Verdict(str, Enum):
    OK = "ok"
    RESTATEMENT = "restatement"  # cites a premise and restates it; valid, no inference
    OK_OVERCITED = "ok_overcited"  # sound, but cited premises the rule did not need
    UNKNOWN_PREMISE = "unknown_premise"  # cited an id that does not exist
    UNKNOWN_RULE = "unknown_rule"
    MISSING_RULE = "missing_rule"  # inference claimed with no rule cited
    ARITY_MISMATCH = "arity_mismatch"  # wrong number of premises for the rule
    ANTECEDENT_MISMATCH = "antecedent_mismatch"  # rule does not fire on cited premises
    NON_SEQUITUR = "non_sequitur"  # rule fires, but yields something else
    MALFORMED = "malformed"  # could not be parsed into a triple at all

    @property
    def is_valid(self) -> bool:
        return self in (Verdict.OK, Verdict.RESTATEMENT, Verdict.OK_OVERCITED)


@dataclass(frozen=True, slots=True)
class Step:
    premise_ids: tuple[str, ...]
    rule_id: str | None
    derived: Atom | None
    raw: str = ""


@dataclass(frozen=True, slots=True)
class StepResult:
    index: int  # 1-based
    verdict: Verdict
    grounded: bool  # every cited premise is itself sound
    detail: str = ""

    @property
    def sound(self) -> bool:
        return self.verdict.is_valid and self.grounded


@dataclass
class DerivationReport:
    steps: list[StepResult]
    theory_size: int

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def first_error_index(self) -> int | None:
        """1-based index of the earliest locally-invalid step, else None."""
        for s in self.steps:
            if not s.verdict.is_valid:
                return s.index
        return None

    @property
    def has_error(self) -> bool:
        return self.first_error_index is not None

    @property
    def n_invalid(self) -> int:
        return sum(1 for s in self.steps if not s.verdict.is_valid)

    @property
    def n_propagated(self) -> int:
        """Steps that are locally valid but rest on an unsound premise.

        This is the direct count of 'reasoned correctly from a wrong thing'.
        """
        return sum(1 for s in self.steps if s.verdict.is_valid and not s.grounded)

    @property
    def all_valid(self) -> bool:
        return self.n_steps > 0 and not self.has_error

    def to_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "first_error_index": self.first_error_index,
            "n_invalid": self.n_invalid,
            "n_propagated": self.n_propagated,
            "all_valid": self.all_valid,
            "verdicts": [s.verdict.value for s in self.steps],
            "grounded": [s.grounded for s in self.steps],
        }


def verify_step(
    theory: Theory,
    available: dict[str, Atom],
    step: Step,
) -> tuple[Verdict, str]:
    """Check one step's *local* validity. Grounding is handled by the caller."""
    if step.derived is None:
        return Verdict.MALFORMED, "no derived proposition"

    premises: list[Atom] = []
    for pid in step.premise_ids:
        if pid in available:
            premises.append(available[pid])
        elif pid in theory.rules:
            # Citing a rule id in the premise slot is a shape error, not a logic
            # error; fold it into the rule slot if none was given.
            if step.rule_id is None:
                return Verdict.MISSING_RULE, f"rule {pid} cited as a premise"
            return Verdict.UNKNOWN_PREMISE, f"{pid} is a rule, not a fact"
        else:
            return Verdict.UNKNOWN_PREMISE, f"no such premise: {pid}"

    if not premises:
        return Verdict.UNKNOWN_PREMISE, "no premises cited"

    # No rule cited: only legitimate as a restatement of something already held.
    if step.rule_id is None:
        if any(p == step.derived for p in premises):
            return Verdict.RESTATEMENT, ""
        return Verdict.MISSING_RULE, "inference claimed with no rule"

    rule = theory.rules.get(step.rule_id)
    if rule is None:
        return Verdict.UNKNOWN_RULE, f"no such rule: {step.rule_id}"

    need = len(rule.antecedents)
    if len(premises) < need:
        return (
            Verdict.ARITY_MISMATCH,
            f"rule needs {need} premise(s), got {len(premises)}",
        )

    if len(premises) == need:
        concl = fire(rule, premises)
        if concl is None:
            return Verdict.ANTECEDENT_MISMATCH, "rule does not fire on the cited premises"
        if concl != step.derived:
            return (
                Verdict.NON_SEQUITUR,
                f"rule yields '{concl.text()}', not '{step.derived.text()}'",
            )
        return Verdict.OK, ""

    # Over-citation: more premises than the rule consumes. This is imprecision,
    # not a reasoning error — a superset of a sufficient premise set still
    # entails the conclusion, which is the pre-registered definition of a
    # correct step. ProofWriter's own gold proofs over-cite (their proof groups
    # carry the inference context, not just the antecedents), so scoring this as
    # an error would flag ~1% of gold proofs as wrong and inflate the headline
    # error rate. Tracked separately so the leniency stays visible.
    if len(premises) > _MAX_OVERCITE:
        return Verdict.ARITY_MISMATCH, f"rule needs {need}, got {len(premises)} (too many to check)"
    fired = [c for sub in combinations(premises, need) if (c := fire(rule, list(sub))) is not None]
    if not fired:
        return Verdict.ANTECEDENT_MISMATCH, "no cited subset satisfies the rule"
    if step.derived in fired:
        return Verdict.OK_OVERCITED, f"cited {len(premises)} premises for a {need}-premise rule"
    return Verdict.NON_SEQUITUR, f"rule yields '{fired[0].text()}', not '{step.derived.text()}'"


def verify_derivation(theory: Theory, steps: list[Step]) -> DerivationReport:
    """Verify a whole chain, tracking local validity and groundedness.

    Derived propositions become citable as `s1`, `s2`, ... in later steps, so a
    model can build on its own work. Whether that work was sound is tracked
    separately — which is exactly how propagation gets measured.
    """
    available: dict[str, Atom] = dict(theory.facts)
    sound_ids: set[str] = set(theory.facts)
    results: list[StepResult] = []

    for i, step in enumerate(steps, start=1):
        verdict, detail = verify_step(theory, available, step)
        grounded = all(pid in sound_ids for pid in step.premise_ids)
        results.append(StepResult(index=i, verdict=verdict, grounded=grounded, detail=detail))

        sid = f"s{i}"
        if step.derived is not None:
            available[sid] = step.derived
            if verdict.is_valid and grounded:
                sound_ids.add(sid)

    return DerivationReport(steps=results, theory_size=len(theory.facts) + len(theory.rules))


def feedback_message(report: DerivationReport, steps: list[Step]) -> str:
    """Machine-generated critique handed back in the verify conditions.

    Reports *what* failed and *where*, never what the answer should be — the
    verifier must not leak the label or it stops being a verifier and becomes
    an oracle.
    """
    bad = [(r, s) for r, s in zip(report.steps, steps, strict=True) if not r.verdict.is_valid]
    if not bad:
        ungrounded = [r for r in report.steps if r.verdict.is_valid and not r.grounded]
        if ungrounded:
            idxs = ", ".join(str(r.index) for r in ungrounded)
            return (
                f"Every step is locally valid, but step(s) {idxs} rely on an earlier "
                "step that was not itself established. Re-derive them from the "
                "numbered premises."
            )
        return ""

    lines = ["The following steps failed verification:"]
    for r, s in bad[:5]:
        lines.append(f"  step {r.index}: {r.verdict.value} — {r.detail or s.raw}")
    lines.append(
        "Revise those steps. Cite only premise ids that exist and rules that actually apply."
    )
    return "\n".join(lines)

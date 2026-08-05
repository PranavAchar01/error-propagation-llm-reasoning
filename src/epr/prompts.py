"""The seven conditions.

Design rule: every condition receives the *identical* numbered premise block
and the identical question. Only the output contract differs. If the structured
conditions saw premise ids that the chain-of-thought conditions did not, any
measured effect would be confounded with information, not format.

Few-shot exemplars are drawn from held-out items of the same dataset and depth
range, never from the evaluation set, and the exemplar chains are the dataset's
own gold proofs rendered into the condition's own output format.
"""

from __future__ import annotations

from dataclasses import dataclass

from .datasets import Item
from .verifier import Step

CONDITIONS = (
    "direct_zs",
    "direct_fs",
    "cot_zs",
    "cot_fs",
    "struct",
    "verify",
    "struct_verify",
)

STRUCTURED = frozenset({"struct", "struct_verify"})
VERIFIED = frozenset({"verify", "struct_verify"})
FEWSHOT = frozenset({"direct_fs", "cot_fs", "struct", "verify", "struct_verify"})


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str


def _answer_instruction(item: Item) -> str:
    space = item.answer_space or ("the letter of the correct option",)
    return f"Answer with exactly one of: {', '.join(space)}."


_TRIPLE_SPEC = """Emit your reasoning as a sequence of steps, one per line, each in exactly this form:

[premises: <ids> | rule: <id> | therefore: <proposition>]

  premises  — the ids of the numbered statements the step uses (comma separated)
  rule      — the id of the rule being applied
  therefore — the single proposition that rule yields, in the same style as the premises

You may cite a previous step's conclusion as s1, s2, ... in later steps.
Cite only ids that appear in the list above. Do not invent ids.
After the final step, write:

ANSWER: <answer>"""

_COT_SPEC = """Reason step by step, then write your final answer on its own last line as:

ANSWER: <answer>"""

_DIRECT_SPEC = """Give only your final answer, on a single line, as:

ANSWER: <answer>"""

SYSTEM = (
    "You are answering formal logic questions. Use only the numbered statements "
    "given. Do not use outside knowledge. Some entities are fictional; that is "
    "intentional and does not change the task."
)


def render_gold_steps(item: Item) -> str:
    """The dataset's own gold proof, in the typed-triple format."""
    return "\n".join(_render_step(s) for s in item.gold_steps)


def _render_step(step: Step) -> str:
    prem = ", ".join(step.premise_ids) if step.premise_ids else "-"
    rule = step.rule_id or "-"
    concl = step.derived.text() if step.derived else "?"
    return f"[premises: {prem} | rule: {rule} | therefore: {concl}]"


def render_gold_chain(item: Item) -> str:
    """The gold proof as prose, for the chain-of-thought conditions."""
    chain = item.meta.get("gold_chain")
    if chain:
        return " ".join(chain)
    return " ".join(s.derived.text() + "." for s in item.gold_steps if s.derived is not None)


def _exemplar(item: Item, condition: str) -> str:
    body = f"Statements:\n{item.premise_block}\n\nQuestion: {item.question}\n\n"
    if condition in STRUCTURED:
        body += render_gold_steps(item) + f"\nANSWER: {item.gold_answer}"
    elif condition in ("cot_fs",) or condition == "verify":
        body += render_gold_chain(item) + f"\nANSWER: {item.gold_answer}"
    else:  # direct_fs
        body += f"ANSWER: {item.gold_answer}"
    return body


def build_prompt(item: Item, condition: str, exemplars: list[Item]) -> Prompt:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")

    if condition in STRUCTURED:
        spec = _TRIPLE_SPEC
    elif condition.startswith("direct"):
        spec = _DIRECT_SPEC
    else:
        spec = _COT_SPEC

    parts: list[str] = []
    if condition in FEWSHOT and exemplars:
        parts.append("Here are worked examples.\n")
        parts.extend(_exemplar(e, condition) + "\n" for e in exemplars)
        parts.append("Now your turn.\n")

    parts.append(f"Statements:\n{item.premise_block}\n")
    parts.append(f"Question: {item.question}\n")
    parts.append(spec.replace("<answer>", "/".join(item.answer_space) or "your answer"))
    parts.append(_answer_instruction(item))

    return Prompt(system=SYSTEM, user="\n".join(parts))


def build_revision_prompt(item: Item, condition: str, prior: str, critique: str) -> Prompt:
    """Second attempt for the verified conditions.

    The critique names which steps failed and why, but never states the answer —
    a verifier that leaks the label is an oracle, and the measurement would be
    of the oracle, not the model.
    """
    spec = _TRIPLE_SPEC if condition in STRUCTURED else _COT_SPEC
    user = (
        f"Statements:\n{item.premise_block}\n\n"
        f"Question: {item.question}\n\n"
        f"Your previous attempt:\n{prior.strip()}\n\n"
        f"An automatic checker found problems:\n{critique}\n\n"
        "Produce a corrected derivation. Do not assume your previous answer was "
        "right or wrong — re-derive it.\n\n"
        + spec.replace("<answer>", "/".join(item.answer_space) or "your answer")
        + "\n"
        + _answer_instruction(item)
    )
    return Prompt(system=SYSTEM, user=user)

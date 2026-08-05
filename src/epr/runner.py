"""The experiment loop: build prompt -> call model -> parse -> verify -> record.

Every record written to results/raw/ is self-describing: it carries the model
version, seed, condition, dataset, item id, the raw response text, and the
verifier's full verdict list. Every table in the paper is regenerable from these
records with no network access.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .datasets import Item
from .model import Client, Response
from .parsers import parse_freeform_steps, parse_structured_steps
from .prompts import STRUCTURED, VERIFIED, Prompt, build_prompt, build_revision_prompt
from .verifier import DerivationReport, Step, feedback_message, verify_derivation

RESULTS = Path(__file__).resolve().parents[2] / "results" / "raw"

_ANSWER = re.compile(r"ANSWER\s*:\s*([A-Za-z()]+)", re.IGNORECASE)


def extract_answer(text: str, answer_space: tuple[str, ...]) -> str | None:
    """Pull the final label. Last match wins — models restate it after revising."""
    matches = _ANSWER.findall(text or "")
    if not matches:
        return None
    raw = matches[-1].strip().strip("().").lower()
    for cand in answer_space:
        if raw == cand.lower():
            return cand
    # Multiple choice (BBH) answers arrive as a bare letter.
    if not answer_space and matches[-1].strip():
        return matches[-1].strip()
    for cand in answer_space:
        if raw.startswith(cand.lower()[:4]):
            return cand
    return None


@dataclass
class Attempt:
    """One model call and everything derived from it."""

    text: str
    answer: str | None
    correct: bool | None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""
    n_steps: int = 0
    parse_failed: bool = False
    n_malformed: int = 0
    report: dict | None = None


@dataclass
class Record:
    uid: str
    dataset: str
    condition: str
    seed: int
    model: str
    depth: int | None
    gold_answer: str
    supports_step_metrics: bool
    first: Attempt
    revised: Attempt | None = None
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _score_derivation(
    item: Item, text: str, condition: str
) -> tuple[list[Step], DerivationReport | None, int, bool]:
    """Parse a response into steps and verify them against the item's theory."""
    if item.theory is None:
        return [], None, 0, False

    if condition in STRUCTURED:
        steps, malformed = parse_structured_steps(text, item.parse_proposition)
    else:
        steps, malformed = parse_freeform_steps(text, item.parse_proposition)

    if not steps:
        return [], None, malformed, True
    return steps, verify_derivation(item.theory, steps), malformed, False


def _run_attempt(
    client: Client, item: Item, condition: str, prompt: Prompt
) -> tuple[Attempt, list[Step], DerivationReport | None]:
    resp: Response = client.complete(prompt.system, prompt.user)
    answer = extract_answer(resp.text, item.answer_space)
    steps, report, malformed, parse_failed = _score_derivation(item, resp.text, condition)
    attempt = Attempt(
        text=resp.text,
        answer=answer,
        correct=None if answer is None else answer == item.gold_answer,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        error=resp.error,
        n_steps=len(steps),
        parse_failed=parse_failed,
        n_malformed=malformed,
        report=report.to_dict() if report else None,
    )
    return attempt, steps, report


def run_item(
    client: Client, item: Item, condition: str, exemplars: list[Item], seed: int
) -> Record:
    """One item under one condition, including the revision pass where it applies."""
    prompt = build_prompt(item, condition, exemplars)
    first, steps, report = _run_attempt(client, item, condition, prompt)

    revised: Attempt | None = None
    if condition in VERIFIED and not first.error:
        critique = ""
        if report is not None:
            critique = feedback_message(report, steps)
        elif first.parse_failed:
            # verification-without-structure: nothing to check. Recording the
            # empty critique is the point of the ablation.
            critique = ""
        if critique:
            rprompt = build_revision_prompt(item, condition, first.text, critique)
            revised, _, _ = _run_attempt(client, item, condition, rprompt)

    return Record(
        uid=item.uid,
        dataset=item.dataset,
        condition=condition,
        seed=seed,
        model=client.model,
        depth=item.depth,
        gold_answer=item.gold_answer,
        supports_step_metrics=item.supports_step_metrics,
        first=first,
        revised=revised,
        meta={"n_gold_steps": len(item.gold_steps)},
    )


def output_path(dataset: str, condition: str, seed: int, phase: str) -> Path:
    p = RESULTS / phase / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{condition}_seed{seed}.jsonl"


def completed_uids(path: Path) -> set[str]:
    """Resume support: never pay twice for the same item."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            done.add(json.loads(line)["uid"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


class BudgetExceeded(RuntimeError):
    """Actual spend crossed the ceiling mid-run. Completed work is already on disk."""


def run_condition(
    client: Client,
    items: list[Item],
    condition: str,
    exemplars: list[Item],
    seed: int,
    phase: str,
    progress=None,
    stop_check=None,
) -> Path:
    """Run one condition, appending each record as soon as it exists.

    `stop_check` is consulted before every item. A pre-run projection is an
    estimate; only real usage can enforce a real ceiling, and because records
    are flushed per item, stopping mid-run loses nothing already paid for —
    `--resume` picks up exactly where it left off.
    """
    path = output_path(items[0].dataset, condition, seed, phase)
    done = completed_uids(path)
    with path.open("a") as fh:
        for item in items:
            if item.uid in done:
                continue
            if stop_check is not None and stop_check():
                raise BudgetExceeded(
                    f"stopped in {items[0].dataset}/{condition} seed={seed}; "
                    f"spend so far ${client.usage.cost(client.model):.2f}"
                )
            rec = run_item(client, item, condition, exemplars, seed)
            fh.write(rec.to_json() + "\n")
            fh.flush()
            if progress:
                progress.update(1)
    return path

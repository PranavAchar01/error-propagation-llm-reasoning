"""The decisive verifier test: it must accept the datasets' own gold proofs.

A verifier that rejects ground truth would manufacture reasoning errors and
inflate every metric in this study. These tests run against committed fixtures
drawn from the real benchmarks, so they are hermetic and require no download.

If one of these fails, no number in the paper can be trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epr.parsers import (
    ParseError,
    UnsupportedTheory,
    build_prontoqa_theory,
    build_proofwriter_theory,
    prontoqa_gold_steps,
    proofwriter_gold_steps,
)
from epr.verifier import verify_derivation

FIXTURES = Path(__file__).parent / "fixtures"


def _prontoqa_items():
    return list(json.loads((FIXTURES / "prontoqa_sample.json").read_text()).items())


def _proofwriter_theories():
    return [
        json.loads(line)
        for line in (FIXTURES / "proofwriter_sample.jsonl").read_text().splitlines()
    ]


# ------------------------------------------------------------------ ProntoQA


@pytest.mark.parametrize("key,ex", _prontoqa_items())
def test_every_prontoqa_gold_chain_verifies_clean(key, ex):
    te = ex["test_example"]
    theory, vocab = build_prontoqa_theory(te["question"])
    steps = prontoqa_gold_steps(te["chain_of_thought"], theory, vocab)
    report = verify_derivation(theory, steps)
    assert report.all_valid, (
        f"{key}: verifier rejected a GOLD chain at step "
        f"{report.first_error_index} — {report.steps[(report.first_error_index or 1) - 1].detail}"
    )
    assert report.n_propagated == 0


def test_prontoqa_gold_chain_length_tracks_hop_count():
    """A k-hop gold chain must linearise to exactly k inferences + 1 restatement."""
    for key, ex in _prontoqa_items():
        hops = int(key.split("_")[0][1:])
        theory, vocab = build_prontoqa_theory(ex["test_example"]["question"])
        steps = prontoqa_gold_steps(ex["test_example"]["chain_of_thought"], theory, vocab)
        assert len(steps) == hops + 1, f"{key}: expected {hops + 1} steps, got {len(steps)}"


def test_prontoqa_parser_is_strict_about_unknown_sentences():
    theory, vocab = build_prontoqa_theory("Every zumpus is a dumpus. Stella is a zumpus.")
    from epr.parsers import parse_prontoqa_sentence

    with pytest.raises(ParseError):
        parse_prontoqa_sentence("Colorless green ideas sleep furiously.", vocab)


# --------------------------------------------------------------- ProofWriter


def test_every_proofwriter_gold_proof_verifies_clean():
    checked = clean = skipped_naf = no_proof = 0
    failures: list[str] = []

    for record in _proofwriter_theories():
        try:
            theory = build_proofwriter_theory(record)
        except UnsupportedTheory:
            skipped_naf += 1
            continue

        for qid, q in record["questions"].items():
            steps = proofwriter_gold_steps(q, theory)
            if not steps:
                no_proof += 1
                assert str(q["answer"]) == "Unknown", (
                    f"{qid}: no gold proof but answer is {q['answer']!r} — only "
                    "open-world Unknown items may lack a proof"
                )
                continue
            checked += 1
            report = verify_derivation(theory, steps)
            if report.all_valid:
                clean += 1
            else:
                bad = report.steps[report.first_error_index - 1]
                failures.append(f"{qid} (QDep={q.get('QDep')}): {bad.verdict.value} — {bad.detail}")

    assert checked >= 90, f"fixture too small to be meaningful (checked {checked})"
    assert not failures, f"{len(failures)}/{checked} gold proofs rejected:\n" + "\n".join(
        failures[:10]
    )
    assert clean == checked
    assert skipped_naf == 2, "fixture should contain exactly the two NAF theories"


def test_proofwriter_gold_proofs_cover_the_full_depth_range():
    depths = set()
    for record in _proofwriter_theories():
        try:
            theory = build_proofwriter_theory(record)
        except UnsupportedTheory:
            continue
        for q in record["questions"].values():
            if proofwriter_gold_steps(q, theory):
                depths.add(q.get("QDep"))
    assert {0, 1, 2, 3, 4, 5} <= depths, f"missing depths: {{0..5}} - {depths}"


def test_negation_as_failure_theories_are_excluded_not_silently_mangled():
    """NAF is unsupported on purpose; it must raise, never parse to a wrong atom."""
    naf = [
        r
        for r in _proofwriter_theories()
        if any("~" in v["representation"] for v in r.get("rules", {}).values())
    ]
    assert naf, "fixture lost its NAF coverage"
    for record in naf:
        with pytest.raises(UnsupportedTheory):
            build_proofwriter_theory(record)


def test_gold_proofs_never_trip_the_overcitation_ceiling():
    """Over-citation is tolerated, but only up to a bound.

    The bound exists so a model cannot pass by citing the whole theory and
    letting subset search find something. Gold proofs must sit well inside it —
    if they did not, the ceiling would be silently rejecting ground truth.
    """
    for record in _proofwriter_theories():
        try:
            theory = build_proofwriter_theory(record)
        except UnsupportedTheory:
            continue
        for q in record["questions"].values():
            for step in proofwriter_gold_steps(q, theory):
                assert len(step.premise_ids) <= 8

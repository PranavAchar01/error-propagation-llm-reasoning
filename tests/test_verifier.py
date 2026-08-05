"""Verifier unit tests: known-valid and known-invalid derivations.

Every error mode the verifier can report has a test that provokes it, and every
test states in its name what a failure would mean for the study.
"""

from __future__ import annotations

import pytest

from epr.logic import Atom, Rule, Theory, fire, unify
from epr.verifier import Step, Verdict, verify_derivation


@pytest.fixture
def theory() -> Theory:
    """Stella is a dumpus; dumpuses are gorpuses; gorpuses are not kind."""
    t = Theory()
    t.facts["fact1"] = Atom("Stella", "is", "dumpus")
    t.facts["fact2"] = Atom("Max", "is", "wumpus")
    t.rules["rule1"] = Rule((Atom("?x", "is", "dumpus"),), Atom("?x", "is", "gorpus"))
    t.rules["rule2"] = Rule((Atom("?x", "is", "gorpus"),), Atom("?x", "is", "kind", False))
    return t


# --------------------------------------------------------------------- valid


def test_valid_two_hop_chain_is_fully_accepted(theory):
    steps = [
        Step(("fact1",), None, Atom("Stella", "is", "dumpus")),
        Step(("s1",), "rule1", Atom("Stella", "is", "gorpus")),
        Step(("s2",), "rule2", Atom("Stella", "is", "kind", False)),
    ]
    r = verify_derivation(theory, steps)
    assert [s.verdict for s in r.steps] == [Verdict.RESTATEMENT, Verdict.OK, Verdict.OK]
    assert r.all_valid and r.first_error_index is None
    assert r.n_propagated == 0
    assert all(s.grounded for s in r.steps)


def test_rule_may_be_applied_directly_to_an_original_fact(theory):
    r = verify_derivation(theory, [Step(("fact1",), "rule1", Atom("Stella", "is", "gorpus"))])
    assert r.all_valid


# ------------------------------------------------------------------- invalid


def test_citing_a_nonexistent_premise_is_caught(theory):
    r = verify_derivation(theory, [Step(("fact99",), "rule1", Atom("Stella", "is", "gorpus"))])
    assert r.steps[0].verdict is Verdict.UNKNOWN_PREMISE
    assert r.first_error_index == 1


def test_citing_a_nonexistent_rule_is_caught(theory):
    r = verify_derivation(theory, [Step(("fact1",), "rule99", Atom("Stella", "is", "gorpus"))])
    assert r.steps[0].verdict is Verdict.UNKNOWN_RULE


def test_rule_that_does_not_apply_to_the_cited_fact_is_caught(theory):
    """Max is a wumpus, and rule1 is about dumpuses. It must not fire."""
    r = verify_derivation(theory, [Step(("fact2",), "rule1", Atom("Max", "is", "gorpus"))])
    assert r.steps[0].verdict is Verdict.ANTECEDENT_MISMATCH


def test_correct_rule_wrong_conclusion_is_a_non_sequitur(theory):
    r = verify_derivation(theory, [Step(("fact1",), "rule1", Atom("Stella", "is", "shumpus"))])
    assert r.steps[0].verdict is Verdict.NON_SEQUITUR
    assert "yields" in r.steps[0].detail


def test_conclusion_with_flipped_polarity_is_a_non_sequitur(theory):
    """The single most important negative case: `is` vs `is not`."""
    r = verify_derivation(
        theory, [Step(("fact1",), "rule1", Atom("Stella", "is", "gorpus", False))]
    )
    assert r.steps[0].verdict is Verdict.NON_SEQUITUR


def test_conclusion_about_the_wrong_entity_is_a_non_sequitur(theory):
    r = verify_derivation(theory, [Step(("fact1",), "rule1", Atom("Max", "is", "gorpus"))])
    assert r.steps[0].verdict is Verdict.NON_SEQUITUR


def test_inference_with_no_rule_cited_is_caught(theory):
    r = verify_derivation(theory, [Step(("fact1",), None, Atom("Stella", "is", "gorpus"))])
    assert r.steps[0].verdict is Verdict.MISSING_RULE


def test_unparseable_conclusion_is_malformed_not_dropped(theory):
    """A step we could not read must still occupy an index in the chain."""
    r = verify_derivation(theory, [Step(("fact1",), "rule1", None)])
    assert r.steps[0].verdict is Verdict.MALFORMED
    assert r.n_steps == 1


def test_arity_mismatch_is_caught():
    t = Theory()
    t.facts["f1"] = Atom("cow", "is", "green")
    t.facts["f2"] = Atom("squirrel", "eats", "cow")
    t.rules["r1"] = Rule(
        (Atom("?x", "is", "green"), Atom("squirrel", "eats", "?x")), Atom("?x", "is", "round")
    )
    r = verify_derivation(t, [Step(("f1",), "r1", Atom("cow", "is", "round"))])
    assert r.steps[0].verdict is Verdict.ARITY_MISMATCH


def test_overcited_step_is_valid_but_flagged(theory):
    """Citing more premises than the rule consumes is imprecision, not error.

    A superset of a sufficient premise set still entails the conclusion, which
    is the pre-registered definition of a correct step. ProofWriter's own gold
    proofs do this, so scoring it as an error would flag ground truth as wrong.
    It still gets its own verdict so the leniency stays measurable.
    """
    r = verify_derivation(
        theory, [Step(("fact1", "fact2"), "rule1", Atom("Stella", "is", "gorpus"))]
    )
    assert r.steps[0].verdict is Verdict.OK_OVERCITED
    assert r.all_valid


def test_overcitation_cannot_be_used_to_brute_force_a_pass(theory):
    """Beyond the ceiling we stop searching subsets, so citing everything fails."""
    r = verify_derivation(
        theory,
        [
            Step(
                tuple(f"fact{i}" for i in range(1, 3)) + ("fact1",) * 9,
                "rule1",
                Atom("Stella", "is", "gorpus"),
            )
        ],
    )
    assert r.steps[0].verdict is Verdict.ARITY_MISMATCH


def test_overcited_step_with_wrong_conclusion_is_still_a_non_sequitur(theory):
    r = verify_derivation(
        theory, [Step(("fact1", "fact2"), "rule1", Atom("Stella", "is", "shumpus"))]
    )
    assert not r.steps[0].verdict.is_valid


def test_multi_antecedent_rule_fires_regardless_of_citation_order():
    """Citing the right premises in the wrong order is not a logical error."""
    t = Theory()
    t.facts["f1"] = Atom("cow", "is", "green")
    t.facts["f2"] = Atom("squirrel", "eats", "cow")
    t.rules["r1"] = Rule(
        (Atom("?x", "is", "green"), Atom("squirrel", "eats", "?x")), Atom("?x", "is", "round")
    )
    for order in (("f1", "f2"), ("f2", "f1")):
        r = verify_derivation(t, [Step(order, "r1", Atom("cow", "is", "round"))])
        assert r.all_valid, f"failed for citation order {order}"


# --------------------------------------------------------------- propagation


def test_error_propagation_is_counted_separately_from_local_validity(theory):
    """The core measurement.

    Step 2 is wrong. Step 3 then reasons *correctly* from step 2's bad output.
    Step 3 must be locally valid but ungrounded — that is what 'propagated'
    means, and conflating it with a fresh error would corrupt the headline
    metric.
    """
    steps = [
        Step(("fact1",), None, Atom("Stella", "is", "dumpus")),
        # Wrong entity: rule1 yields "Stella is gorpus", not "Max is gorpus".
        # Chosen so the bad output still satisfies rule2's antecedent, which is
        # what lets step 3 be locally valid while resting on an error.
        Step(("s1",), "rule1", Atom("Max", "is", "gorpus")),
        Step(("s2",), "rule2", Atom("Max", "is", "kind", False)),  # valid *given* s2
    ]
    r = verify_derivation(theory, steps)
    assert r.first_error_index == 2
    assert r.steps[1].verdict is Verdict.NON_SEQUITUR
    assert r.steps[2].verdict is Verdict.OK  # locally sound reasoning...
    # step 3 cites s2, which is unsound -> locally checked, but not grounded
    assert r.steps[2].grounded is False
    assert r.n_propagated >= 1
    assert r.n_invalid == 1


def test_first_error_index_reports_the_earliest_not_the_worst(theory):
    steps = [
        Step(("fact99",), "rule1", Atom("Stella", "is", "gorpus")),  # error at 1
        Step(("fact1",), "rule99", Atom("Stella", "is", "gorpus")),  # error at 2
    ]
    r = verify_derivation(theory, steps)
    assert r.first_error_index == 1
    assert r.n_invalid == 2


def test_a_clean_chain_after_a_bad_start_is_still_ungrounded(theory):
    """Recovering locally does not launder an unsound premise."""
    steps = [
        Step(("fact2",), "rule1", Atom("Max", "is", "gorpus")),  # rule does not apply
        Step(("s1",), "rule2", Atom("Max", "is", "kind", False)),
    ]
    r = verify_derivation(theory, steps)
    assert r.steps[0].verdict is Verdict.ANTECEDENT_MISMATCH
    assert r.steps[1].verdict is Verdict.OK
    assert r.steps[1].grounded is False
    assert r.steps[1].sound is False


# ------------------------------------------------------------------ unify/fire


def test_unify_respects_polarity_and_relation():
    assert unify(Atom("?x", "is", "green"), Atom("cow", "is", "green")) == {"?x": "cow"}
    assert unify(Atom("?x", "is", "green"), Atom("cow", "is", "green", False)) is None
    assert unify(Atom("?x", "is", "green"), Atom("cow", "eats", "green")) is None


def test_unify_binds_a_variable_consistently_across_slots():
    rule = Rule((Atom("?x", "is", "green"), Atom("?x", "eats", "tiger")), Atom("?x", "is", "round"))
    assert fire(rule, [Atom("cow", "is", "green"), Atom("cow", "eats", "tiger")]) == Atom(
        "cow", "is", "round"
    )
    # different entities must not satisfy a rule quantified over one variable
    assert fire(rule, [Atom("cow", "is", "green"), Atom("dog", "eats", "tiger")]) is None


def test_rule_with_unbound_consequent_variable_yields_nothing():
    rule = Rule((Atom("?x", "is", "green"),), Atom("someone", "is", "round"))
    assert fire(rule, [Atom("cow", "is", "green")]) is None

"""Propositional/unary-FOL core shared by ProntoQA and ProofWriter.

Deliberately small. Everything here is pure data plus unification; there is no
model anywhere in this module and there never should be. The verifier's whole
value is that it is mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

# Free variables. ProofWriter writes them as English indefinites; ProntoQA rules
# are universally quantified over a single variable we name explicitly.
VARIABLES = frozenset({"something", "someone", "it", "?x"})


def is_var(token: str) -> bool:
    return token.lower() in VARIABLES


@dataclass(frozen=True, slots=True)
class Atom:
    """A ground or patterned literal: (subject, relation, object, polarity).

    ProntoQA uses relation="is" throughout. ProofWriter also uses "eats",
    "sees", "chases", "likes", "needs", "visits".
    """

    subject: str
    relation: str
    object: str
    polarity: bool = True

    def text(self) -> str:
        neg = "" if self.polarity else "not "
        return f"{self.subject} {self.relation} {neg}{self.object}"

    def negated(self) -> Atom:
        return Atom(self.subject, self.relation, self.object, not self.polarity)


@dataclass(frozen=True, slots=True)
class Rule:
    """antecedents -> consequent, universally closed over any variables."""

    antecedents: tuple[Atom, ...]
    consequent: Atom

    def text(self) -> str:
        return " and ".join(a.text() for a in self.antecedents) + " -> " + self.consequent.text()


@dataclass
class Theory:
    """The premise set a derivation is checked against."""

    facts: dict[str, Atom] = field(default_factory=dict)
    rules: dict[str, Rule] = field(default_factory=dict)

    def has(self, pid: str) -> bool:
        return pid in self.facts or pid in self.rules


Binding = dict[str, str]


def _bind(pattern: str, ground: str, binding: Binding) -> Binding | None:
    """Bind one slot. Constants must match exactly; variables bind consistently."""
    if is_var(pattern):
        key = pattern.lower()
        if key in binding:
            return binding if binding[key] == ground else None
        out = dict(binding)
        out[key] = ground
        return out
    return binding if pattern == ground else None


def unify(pattern: Atom, ground: Atom, binding: Binding | None = None) -> Binding | None:
    """Match a rule antecedent against a concrete atom.

    Polarity and relation must match exactly — a rule about `is not green` does
    not fire on `is green`. That strictness is the point.
    """
    if binding is None:
        binding = {}
    if pattern.relation != ground.relation or pattern.polarity != ground.polarity:
        return None
    b = _bind(pattern.subject, ground.subject, binding)
    if b is None:
        return None
    return _bind(pattern.object, ground.object, b)


def substitute(atom: Atom, binding: Binding) -> Atom:
    """Instantiate a patterned atom under a binding."""
    subj = binding.get(atom.subject.lower(), atom.subject) if is_var(atom.subject) else atom.subject
    obj = binding.get(atom.object.lower(), atom.object) if is_var(atom.object) else atom.object
    return Atom(subj, atom.relation, obj, atom.polarity)


def fire(rule: Rule, premises: list[Atom]) -> Atom | None:
    """Return the conclusion of `rule` given exactly `premises`, or None.

    Premises are matched to antecedents in any order — a model that cites the
    right two facts in the wrong order has not made a logical error. Arity must
    match exactly: citing extra or missing premises is a real error and is
    reported as one by the caller.
    """
    if len(premises) != len(rule.antecedents):
        return None
    for perm in permutations(premises):
        binding: Binding | None = {}
        for ant, prem in zip(rule.antecedents, perm, strict=True):
            binding = unify(ant, prem, binding)
            if binding is None:
                break
        if binding is not None:
            concl = substitute(rule.consequent, binding)
            # A consequent with unbound variables is not a usable conclusion.
            if not is_var(concl.subject) and not is_var(concl.object):
                return concl
    return None

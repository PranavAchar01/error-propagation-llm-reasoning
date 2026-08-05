"""Dataset-native parsers: benchmark text -> Theory, and model text -> Steps.

ProntoQA states its premises in a tiny, fully enumerable English grammar.
ProofWriter ships a machine-readable s-expression alongside every sentence.
Neither needs a model to read, which is what keeps the verifier honest.

Parsers here are STRICT: unrecognised input raises rather than silently
producing a wrong atom. A silent mis-parse would show up as a fake reasoning
error and inflate the very metric this study reports.
"""

from __future__ import annotations

import re

from .logic import Atom, Rule, Theory
from .verifier import Step


class ParseError(ValueError):
    """Raised when input does not match the dataset's own grammar."""


class UnsupportedTheory(ParseError):
    """The theory uses a construct this verifier cannot soundly check.

    Currently only ProofWriter's `~` polarity — negation as failure. A `~`
    literal asserts *unprovability*, which cannot be decided by checking a step
    against its cited premises: it requires a closed-world search over the whole
    theory. Rather than approximate it and silently mis-score steps, theories
    containing NAF are excluded and counted. The exclusion rate is reported.
    """


# --------------------------------------------------------------------------
# ProntoQA
# --------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=\.)\s+")
_UNIVERSAL = re.compile(r"^(?:Every|Each)\s+(\w+)\s+is\s+(not\s+)?(?:an?\s+)?(\w+)\.?$", re.IGNORECASE)
_PLURAL_RULE = re.compile(r"^(\w+)\s+are\s+(not\s+)?(\w+)\.?$", re.IGNORECASE)
_GROUND = re.compile(r"^([A-Z]\w*)\s+is\s+(not\s+)?(?:an?\s+)?(\w+)\.?$")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


class Vocab:
    """Maps plural surface forms to the singular predicate name.

    Built from positions where ProntoQA is unambiguously singular ("Every X",
    "is a Y"), then used to resolve the plural positions ("Xs are Ys"). Doing it
    from the data rather than from an English pluralisation rule means the
    fictional ontology (`zumpus`/`zumpuses`) and any real-word ontology both
    work without special-casing.
    """

    def __init__(self) -> None:
        self.singulars: set[str] = set()

    def observe(self, word: str) -> None:
        self.singulars.add(word.lower())

    def singularize(self, word: str) -> str:
        w = word.lower()
        if w in self.singulars:
            return w
        for cand in (w[:-2], w[:-1]):  # "zumpuses" -> "zumpus"; "cats" -> "cat"
            if cand in self.singulars:
                return cand
        # Unseen plural: fall back to the -uses convention, which is idempotent
        # for already-singular -us nouns.
        if w.endswith("uses"):
            return w[:-2]
        return w


def _scan_vocab(sentences: list[str]) -> Vocab:
    v = Vocab()
    for s in sentences:
        if m := _UNIVERSAL.match(s):
            v.observe(m.group(1))
            v.observe(m.group(3))
        elif m := _GROUND.match(s):
            v.observe(m.group(3))
    return v


def parse_prontoqa_sentence(sentence: str, vocab: Vocab) -> Atom | Rule:
    """One ProntoQA sentence -> a ground Atom or a universal Rule."""
    s = sentence.strip()

    if m := _UNIVERSAL.match(s):
        subj, neg, obj = m.group(1), m.group(2), m.group(3)
        return Rule(
            antecedents=(Atom("?x", "is", vocab.singularize(subj), True),),
            consequent=Atom("?x", "is", vocab.singularize(obj), neg is None),
        )

    if m := _GROUND.match(s):
        name, neg, obj = m.group(1), m.group(2), m.group(3)
        return Atom(name, "is", vocab.singularize(obj), neg is None)

    if m := _PLURAL_RULE.match(s):
        subj, neg, obj = m.group(1), m.group(2), m.group(3)
        return Rule(
            antecedents=(Atom("?x", "is", vocab.singularize(subj), True),),
            consequent=Atom("?x", "is", vocab.singularize(obj), neg is None),
        )

    raise ParseError(f"unrecognised ProntoQA sentence: {sentence!r}")


def build_prontoqa_theory(question: str) -> tuple[Theory, Vocab]:
    """Premise text -> Theory with stable ids (fact1.., rule1..)."""
    sentences = _split_sentences(question)
    vocab = _scan_vocab(sentences)
    theory = Theory()
    nf = nr = 0
    for s in sentences:
        parsed = parse_prontoqa_sentence(s, vocab)
        if isinstance(parsed, Atom):
            nf += 1
            theory.facts[f"fact{nf}"] = parsed
            theory.texts[f"fact{nf}"] = s
        else:
            nr += 1
            theory.rules[f"rule{nr}"] = parsed
            theory.texts[f"rule{nr}"] = s
    return theory, vocab


def prontoqa_gold_steps(chain: list[str], theory: Theory, vocab: Vocab) -> list[Step]:
    """Gold `chain_of_thought` -> typed triples.

    ProntoQA chains alternate: a grounding fact, then (rule, conclusion) pairs.
    Converting the gold chain into the same triple format the model must emit is
    what lets the verifier be validated against the dataset's own proofs.
    """
    fact_id = {a: i for i, a in theory.facts.items()}
    rule_id = {r: i for i, r in theory.rules.items()}

    steps: list[Step] = []
    parsed = [parse_prontoqa_sentence(s, vocab) for s in chain]

    if not parsed or not isinstance(parsed[0], Atom):
        raise ParseError("gold chain does not start with a ground fact")

    first = parsed[0]
    if first not in fact_id:
        raise ParseError(f"gold chain opens with a fact not in the theory: {first.text()}")
    steps.append(Step((fact_id[first],), None, first, chain[0]))

    i = 1
    while i + 1 < len(parsed):
        rule, concl = parsed[i], parsed[i + 1]
        if not isinstance(rule, Rule) or not isinstance(concl, Atom):
            raise ParseError(f"expected (rule, conclusion) at chain index {i}")
        if rule not in rule_id:
            raise ParseError(f"gold chain cites a rule not in the theory: {rule.text()}")
        steps.append(Step((f"s{len(steps)}",), rule_id[rule], concl, chain[i + 1]))
        i += 2

    return steps


# --------------------------------------------------------------------------
# ProofWriter
# --------------------------------------------------------------------------

_PW_ATOM = re.compile(r'\(\s*"([^"]*)"\s+"([^"]*)"\s+"([^"]*)"\s+"([+~-])"\s*\)')


def parse_pw_atom(rep: str) -> Atom:
    m = _PW_ATOM.search(rep)
    if not m:
        raise ParseError(f"unrecognised ProofWriter atom: {rep!r}")
    if m.group(4) == "~":
        raise UnsupportedTheory(f"negation as failure is not locally checkable: {rep!r}")
    return Atom(m.group(1), m.group(2), m.group(3), m.group(4) == "+")


def parse_pw_rule(rep: str) -> Rule:
    if "->" not in rep:
        raise ParseError(f"ProofWriter rule has no implication: {rep!r}")
    lhs, rhs = rep.split("->", 1)
    found = _PW_ATOM.findall(lhs)
    if any(p == "~" for *_, p in found):
        raise UnsupportedTheory(f"rule antecedent uses negation as failure: {rep!r}")
    ants = tuple(Atom(a, r, o, p == "+") for a, r, o, p in found)
    if not ants:
        raise ParseError(f"ProofWriter rule has no antecedents: {rep!r}")
    return Rule(antecedents=ants, consequent=parse_pw_atom(rhs))


def build_proofwriter_theory(record: dict) -> Theory:
    """A ProofWriter theory record -> Theory, preserving the dataset's own ids.

    Keeping `triple3`/`rule7` as the ids means the model cites exactly what the
    gold `proofs` field cites, so gold proofs are checkable without remapping.
    """
    theory = Theory()
    for tid, t in record.get("triples", {}).items():
        theory.facts[tid] = parse_pw_atom(t["representation"])
        theory.texts[tid] = t["text"]
    for rid, r in record.get("rules", {}).items():
        theory.rules[rid] = parse_pw_rule(r["representation"])
        theory.texts[rid] = r["text"]
    return theory


_PROOF_TOK = re.compile(r"\(|\)|->|%|[A-Za-z]+\d*")


def _parse_proof_node(
    toks: list[str], pos: int, inters: dict, out: list[Step], seen: dict[str, str]
) -> tuple[list[str], int]:
    """Recursive descent over one ProofWriter proof node.

    Grammar (inferred from the corpus, then validated against every gold proof):
        node := '(' leaf ')'                                  -- a cited fact
              | '(' node+ '->' '(' RULE '%' INT ')' ')'        -- a rule application
              | '(' node+ ')'                                  -- an antecedent group

    The third form is why this returns a *list*: a multi-antecedent rule wraps
    its antecedents in a group, so a node can yield more than one premise id.
    Sub-proofs are shared by intermediate id, so a lemma used twice is emitted
    once and cited twice — the linearised chain is a DAG walk, not a tree
    expansion with duplicated work.
    """
    if toks[pos] != "(":
        raise ParseError(f"expected '(' at {pos}")
    pos += 1
    children: list[str] = []
    while pos < len(toks) and toks[pos] not in (")", "->"):
        if toks[pos] == "(":
            cids, pos = _parse_proof_node(toks, pos, inters, out, seen)
            children.extend(cids)
        else:
            children.append(toks[pos])
            pos += 1

    if pos < len(toks) and toks[pos] == "->":
        pos += 1
        if toks[pos] != "(":
            raise ParseError("expected '(' after '->'")
        pos += 1
        rule_id = toks[pos]
        pos += 1
        if toks[pos] != "%":
            raise ParseError("expected '%' in rule application")
        pos += 1
        int_id = toks[pos]
        pos += 1
        for _ in range(2):  # close the (rule % int) group and the node
            if pos < len(toks) and toks[pos] == ")":
                pos += 1
        if int_id not in seen:
            rep = inters.get(int_id, {}).get("representation")
            if rep is None:
                raise ParseError(f"proof cites unknown intermediate {int_id}")
            out.append(
                Step(
                    tuple(children),
                    rule_id,
                    parse_pw_atom(rep),
                    f"{children} -{rule_id}-> {int_id}",
                )
            )
            seen[int_id] = f"s{len(out)}"
        return [seen[int_id]], pos

    if pos < len(toks) and toks[pos] == ")":
        pos += 1
    if not children:
        raise ParseError("empty proof group")
    return children, pos


def proofwriter_gold_steps(question: dict, theory: Theory | None = None) -> list[Step]:
    """A ProofWriter question's gold proof -> typed triples in dependency order.

    Depth-0 questions are answered by a bare fact and produce a single
    restatement step. Questions answered `Unknown` under the open-world
    assumption have no positive proof and produce no steps — they are excluded
    from step-level metrics rather than scored as zero-length successes.
    """
    pwi = question.get("proofsWithIntermediates") or []
    if not pwi:
        return []
    rep = pwi[0].get("representation", "")
    inters = pwi[0].get("intermediates", {}) or {}

    # Depth-0: the answer is a stated fact, so the proof is the fact id alone.
    # Resolving it against the theory turns it into a checkable restatement
    # instead of a step with nothing derived.
    bare = re.fullmatch(r"\(*\s*(triple\d+)\s*\)*", rep.strip())
    if bare:
        tid = bare.group(1)
        atom = theory.facts.get(tid) if theory else None
        if atom is None:
            raise ParseError(f"depth-0 proof cites {tid}, absent from the theory")
        return [Step((tid,), None, atom, rep)]

    toks = _PROOF_TOK.findall(rep)
    out: list[Step] = []
    _parse_proof_node(toks, 0, inters, out, {})
    return out


# --------------------------------------------------------------------------
# Model output -> Steps
# --------------------------------------------------------------------------

# [premises: fact1, fact2 | rule: rule3 | therefore: Stella is not kind]
_TRIPLE = re.compile(
    r"\[\s*premises?\s*:\s*([^|\]]*)\|\s*rule\s*:\s*([^|\]]*)\|\s*therefore\s*:\s*([^\]]*)\]",
    re.IGNORECASE,
)
_ID = re.compile(r"\b((?:fact|rule|triple|s)\d+)\b", re.IGNORECASE)


def parse_structured_steps(
    text: str, prop_parser, *, max_steps: int = 64
) -> tuple[list[Step], int]:
    """Extract typed triples from a structured response.

    Returns (steps, n_malformed). `prop_parser` turns the free-text conclusion
    into an Atom and is dataset-specific; a conclusion that will not parse
    yields a MALFORMED step rather than being dropped, so the denominator stays
    honest.
    """
    steps: list[Step] = []
    malformed = 0
    for m in _TRIPLE.finditer(text):
        if len(steps) >= max_steps:
            break
        prem_raw, rule_raw, concl_raw = m.group(1), m.group(2), m.group(3)
        prem_ids = tuple(pid.lower() for pid in _ID.findall(prem_raw))
        rule_ids = _ID.findall(rule_raw)
        rule_id = rule_ids[0].lower() if rule_ids else None
        try:
            derived = prop_parser(concl_raw.strip())
        except (ParseError, ValueError):
            derived = None
            malformed += 1
        steps.append(Step(prem_ids, rule_id, derived, m.group(0)))
    return steps, malformed


def parse_freeform_steps(text: str, prop_parser) -> tuple[list[Step], int]:
    """Best-effort recovery of a derivation from unstructured chain-of-thought.

    Used only by the `verify` (verification-without-structure) condition. It
    cannot recover premise ids, because free prose does not cite them — which is
    precisely the finding that ablation is designed to expose. The resulting
    parse-failure rate is reported, not hidden.
    """
    steps: list[Step] = []
    malformed = 0
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if not line or len(line) < 4:
            continue
        try:
            parsed = prop_parser(line)
        except (ParseError, ValueError):
            continue
        if isinstance(parsed, Atom):
            steps.append(Step((), None, parsed, line))
        else:
            malformed += 1
    return steps, malformed

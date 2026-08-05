# Where Reasoning Breaks: Error Propagation in Multi-Step LLM Deduction

Does forcing a language model to commit to explicit, machine-checkable
intermediate steps improve multi-step logical reasoning — and does it reduce
**error propagation**, the tendency for one wrong intermediate step to corrupt
every step after it?

*Independent research project. Not affiliated with, endorsed by, or conducted
under any research lab.*

---

## Status

**The apparatus is complete and validated. The experiment has not been run.**

Execution needs an `OPENAI_API_KEY`, which was not available in the build
environment. Everything up to the first API call is done, tested, and
reproducible. `paper/report.md` marks every results section `[PENDING]`; there
are no placeholder numbers anywhere that could be mistaken for findings.

| stage | state |
|---|---|
| Pre-registration (`HYPOTHESIS.md`) | done — committed before any model call |
| Benchmarks downloaded, generated, checksummed | done |
| Non-model verifier + calibration | done — **100% on 4,832 gold proofs** |
| Harness, 7 conditions, cost control | done — dry-runs clean |
| Metrics + statistics pipeline | done — validated on a planted effect |
| 50-item pilot (~$0.76) | **blocked: needs API key** |
| Full run (~$24.68, fits the $25 ceiling) | **blocked: needs API key** |
| Report results sections | blocked on the above |

## The question, precisely

"Chain-of-thought helps" is established and is included here only as a
manipulation check. The contribution is the second half: when a model produces a
ten-step derivation and gets the answer wrong, did it fail at step 9, or fail at
step 2 and then reason impeccably from a false premise for eight more steps?
Those look identical from outside — one wrong answer — and demand different
fixes. The verifier tracks two things separately to tell them apart:

- **local validity** — is the step sound *treating its cited premises as given*?
- **groundedness** — are those premises themselves traceable to the theory?

A step that is locally valid but *not* grounded is a propagated error: correct
reasoning from an already-wrong premise.

**Primary metric (pre-registered):** `Δβ = β_depth(struct_verify) − β_depth(cot_fs)`
on PrOntoQA, where `β_depth` is the slope of `logit P(correct)` on ground-truth
proof depth. A flatter slope is the win condition.

## The verifier is not a model

A step is a typed triple:

```
[premises: fact2, s1 | rule: rule7 | therefore: Stella is not kind]
```

~200 lines of Python resolve the cited ids, unify the rule's antecedents against
the cited premises, and check the conclusion is exactly what the rule yields.
A step citing premises that do not entail it is a *detectable* error, not a
stylistic one.

**It is calibrated against ground truth**, because a verifier that rejects gold
proofs would manufacture errors and inflate every metric here:

| dataset | gold proofs checked | verified clean |
|---|---|---|
| PrOntoQA (hops 1–5) | 200 | **100.00%** |
| ProofWriter OWA depth-5 (QDep 0–5) | 4,632 | **100.00%** |

Two corrections were needed to get there, and both are findings about the
benchmarks rather than incidental bugs:

- **Over-citation is not an error.** ProofWriter's proof groups carry inference
  *context*, not just antecedents, so ~1% of gold proofs cite more premises than
  the rule consumes. A superset still entails the conclusion, so this is now
  valid-but-flagged — bounded at 8 premises so a model cannot brute-force a pass.
- **Negation-as-failure is not locally checkable.** ~3.5% of ProofWriter theories
  use `~` (unprovability), which needs a closed-world search. Those theories are
  excluded and **counted**, never approximated.

## Design

Seven conditions, all seeing the **identical** numbered premise block — only the
output contract differs, so an effect cannot be confounded with information:

`direct_zs` · `direct_fs` · `cot_zs` · `cot_fs` · `struct` · `verify` · `struct_verify`

`struct` isolates structure without feedback; `verify` applies the verifier to a
best-effort parse of free prose, isolating whether the benefit comes from having
a checker or from being *checkable at all*. Its parse-failure rate is a reported
result, not a hidden loss.

| dataset | depth knob | gold steps | supports β_depth | role |
|---|---|---|---|---|
| PrOntoQA (fictional) | exact hop count | yes | **yes** | primary |
| ProofWriter OWA depth-5 | `QDep` 0–5 | yes | **yes** | confirmatory |
| FOLIO | none | **no** | **no** | generalisation |
| BBH `logical_deduction` | object count (proxy) | no | no | OOD check |

FOLIO has no gold proof chain, so first-error position and recovery rate are
**blank** there — never estimated. Producing them would require a model to grade
itself.

## Run it

```bash
make setup      # pinned deps, Python 3.12
make data       # download + generate + checksum every benchmark
make test       # 112 tests, no network, no API key
make check      # lint + test

make auth-check # verify the key + confirm the model exists (1 tiny call)
make pilot      # 50 items/condition, projects ~$0.76   (needs OPENAI_API_KEY)
make full       # full grid, projects ~$24.68          (needs OPENAI_API_KEY)
make reproduce  # regenerate every table from results/raw/, NO network
```

Any phase can be costed without spending anything:

```bash
.venv/bin/python scripts/run.py --phase full --n 290 --seeds 1 2 3 --dry-run
```

Runs refuse to *start* above a `--max-cost` ceiling (default $25), and are also
checked against **actual** spend before every item — a projection is an estimate,
so only real usage can enforce a real ceiling. Records flush per item, so a
budget stop loses nothing already paid for and a re-run resumes from it.

Set credentials in `.env` (see `.env.example`); it is loaded automatically.
The subject model is **`gpt-4.1-mini-2025-04-14`** — see Amendments 1 and 2 in
[HYPOTHESIS.md](HYPOTHESIS.md) for why it changed and why that does not touch the
pre-registered metric.

## Honesty commitments

Taken before any data existed, and enforced by the code:

- If the intervention does not help, **the abstract says so in bold, in the first
  two sentences.**
- Every condition run is reported, including failures. No quiet dropping.
- Every percentage carries its **n**. Unmeasurable quantities render as `--`,
  never `0`.
- A direct-answer baseline above 95% is auto-flagged as possible contamination.
- A non-significant primary result is reported as *"no detected effect, MDE = X"*,
  never *"no effect"* — the minimum detectable effect is computed by simulation.
  If the MDE exceeds the pre-registered prediction, the study declares itself
  underpowered for its own hypothesis.
- No number appears that cannot be regenerated from `results/raw/` offline.

## Layout

```
HYPOTHESIS.md     pre-registration, frozen (committed before any model call)
src/epr/
  logic.py        atoms, rules, unification
  verifier.py     the non-model verifier
  parsers.py      benchmark grammars + model-output parsing
  datasets.py     uniform Item loader; every dropped item counted
  prompts.py      the seven conditions
  model.py        provider-agnostic client (OpenAI/Anthropic), retries, cost
  runner.py       experiment loop, resumable
  metrics.py      pre-registered metrics, denominators tracked
  stats.py        bootstrap, McNemar, Holm-Bonferroni, power
scripts/          prepare_data · auth_check · run · analyze
paper/report.md   method, threats to validity, limitations
data/README.md    provenance, licences, known quirks
```

## License

MIT for code. Benchmarks retain their original licences — see
[data/README.md](data/README.md).

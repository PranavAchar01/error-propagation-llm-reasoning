# Where Reasoning Breaks: Measuring Error Propagation in Multi-Step LLM Deduction

*Independent research project. Not affiliated with, endorsed by, or conducted under any research lab.*

---

## Abstract

> **The intervention did not work. On the one benchmark able to measure it,
> forcing a model to emit machine-checkable intermediate steps made error
> propagation significantly *worse*, not better: Δβ = −0.174 (95% CI −0.276 to
> −0.077, p = 0.0008, Holm-adjusted p = 0.0024, n = 865 paired items), where the
> pre-registered prediction was Δβ = +0.15 in the opposite direction.**
>
> **The pre-registered primary comparison could not be run at all.** PrOntoQA,
> the designated primary instrument, turned out to be saturated for this model —
> answering with no reasoning whatsoever scores 98.9% — so β_depth ≈ 0 in every
> condition and C1 is reported as *not estimable*, never as a null.

Chain-of-thought prompting improves multi-step reasoning accuracy; this is
established and is included here only as a manipulation check. This study asks a
different question: *where in a chain does reasoning break, and does forcing a
model to emit machine-checkable intermediate steps change the shape of that
failure curve?* We operationalise **error propagation** as the slope of accuracy
against ground-truth proof depth, and measure it on two benchmarks with
controllable depth and gold proof annotations (PrOntoQA, ProofWriter), with two
further benchmarks as generalisation and out-of-distribution checks (FOLIO,
BIG-Bench Hard `logical_deduction`). The intervention requires each step to be a
typed triple — cited premise ids, cited rule, derived proposition — that a
**Python verifier containing no model** checks against the premise set. Ablations
separate structure from verification.

Primary metric, sample sizes, predictions, and the stopping rule were
pre-registered in `HYPOTHESIS.md` and committed to git **before** any model call
(commit `4325878`, which precedes every commit touching `results/`).

---

## 1. Introduction

"Chain-of-thought helps" is not worth re-deriving. The interesting and
under-measured question is *structural*: when a model produces a ten-step
derivation and gets the answer wrong, did it fail at step 9, or did it fail at
step 2 and then reason impeccably from a false premise for eight more steps?

Those two failures look identical from the outside — one wrong answer — and they
call for completely different interventions. They are also an accountability
question, not merely an accuracy one: a system that reasons validly from a
corrupted intermediate is *more* dangerous than one that reasons sloppily
throughout, because its output is internally coherent and its error is invisible
to any check that reads only the conclusion.

This study makes three commitments that distinguish it from a prompt-comparison
exercise:

1. **Error position is measured, not inferred.** Step-level correctness is scored
   against each dataset's own proof annotations. No model grades an unlabelled
   chain.
2. **The checker is not a model.** The verifier is ~200 lines of Python doing
   unification and premise resolution. It has no opinions.
3. **The failure mode is pre-registered.** The prediction that the intervention
   *lowers* post-error recovery — which reads as a regression unless you know
   what recovery measures — is committed in advance so it cannot be reframed
   after the fact.

---

## 2. Method

### 2.1 Operational definitions

The core of the design is a distinction the verifier tracks explicitly:

- **Local validity.** Is the step's inference sound *treating its cited premises
  as given*? A step citing `fact3` and `rule7` is locally valid iff `rule7`
  actually fires on `fact3` and yields exactly the stated conclusion.
- **Groundedness.** Are those cited premises themselves traceable to the original
  theory through a chain of locally-valid steps?

A step that is **locally valid but not grounded** is a propagated error: the model
reasoned correctly from something it had already gotten wrong. Conflating this
with a fresh error would corrupt the headline metric, which is why the two are
counted separately.

From these, the pre-registered metrics:

| metric | definition | denominator |
|---|---|---|
| **first-error position** | 1-based index of the earliest locally-invalid step | items with ≥1 invalid step |
| **post-error recovery rate** | reached the correct answer *despite* a broken chain | items with ≥1 invalid step |
| **depth-conditioned accuracy** | accuracy as a function of gold proof depth | all items, binned |
| **β_depth** | slope of `logit P(correct)` on depth | the propagation number |
| **verifier rejection rate** | fraction of emitted steps rejected | all emitted steps |
| **parse-failure rate** | fraction of responses yielding no derivation | all items |

**Primary metric (pre-registered):** `Δβ = β_depth(struct_verify) − β_depth(cot_fs)`
on PrOntoQA. β_depth is negative by construction; a *flatter* slope under the
intervention (Δβ > 0) is the win condition.

**On recovery rate.** High recovery is a *bad* sign, not a good one: it means the
model reached the right answer while its own stated chain was broken, i.e. the
chain was decorative rather than causal. The pre-registered prediction is that
the intervention **lowers** recovery while also lowering error frequency. Δβ and
recovery may therefore move in opposite directions, and that combination is a
positive result for faithfulness. This is stated in `HYPOTHESIS.md` §3 in
advance precisely so it cannot be retrofitted.

### 2.2 The verifier

A step is a typed triple:

```
[premises: fact2, s1 | rule: rule7 | therefore: Stella is not kind]
```

The verifier resolves the cited ids against the premise set, unifies the cited
rule's antecedents against the cited premises, and checks that the stated
conclusion is exactly what the rule yields. Verdicts are: `ok`, `restatement`,
`ok_overcited`, `unknown_premise`, `unknown_rule`, `missing_rule`,
`arity_mismatch`, `antecedent_mismatch`, `non_sequitur`, `malformed`.

**Calibration against ground truth.** A verifier that rejects gold proofs would
manufacture reasoning errors and inflate every metric in this study. Measured on
the datasets' own annotations:

| dataset | gold proofs checked | verified clean |
|---|---|---|
| PrOntoQA (hops 1–5) | 200 | **200 / 200 = 100.00%** |
| ProofWriter OWA depth-5 (QDep 0–5) | 4,632 | **4,632 / 4,632 = 100.00%** |

Reaching 100% required two corrections, both of which are findings about the
benchmarks rather than incidental bugs:

- **Over-citation is not an error.** ProofWriter's proof groups record the
  inference *context*, not strictly a rule's antecedents, so ~1% of gold proofs
  cite more premises than the rule consumes. A strict arity check flagged ground
  truth as wrong. Since a superset of a sufficient premise set still entails the
  conclusion — the pre-registered definition of a correct step — over-citation is
  now valid-but-flagged, and bounded at 8 cited premises so a model cannot
  brute-force a pass by citing the whole theory.
- **Negation as failure is not locally checkable.** ~3.5% of ProofWriter theories
  use the `~` polarity marker, which asserts *unprovability*. Deciding it needs a
  closed-world search over the entire theory, not a check of a step against its
  cited premises. Those theories are **excluded and counted**, never approximated.

### 2.3 Conditions

Seven conditions, `k=4` few-shot exemplars drawn from held-out items:

| id | structure | verifier | description |
|---|---|---|---|
| `direct_zs` / `direct_fs` | – | – | answer only |
| `cot_zs` / `cot_fs` | – | – | free-form chain-of-thought |
| `struct` | ✓ | – | typed triples, no feedback |
| `verify` | – | ✓ | free-form CoT, best-effort parse, feedback |
| `struct_verify` | ✓ | ✓ | typed triples + feedback |

**Every condition receives the identical numbered premise block and question.**
Only the output contract differs. Had the structured conditions seen premise ids
that the CoT conditions did not, any measured effect would be confounded with
information rather than format.

`verify` is the ablation that isolates whether the benefit comes from *having a
checker* or from *being checkable at all*: the same verifier is applied to a
best-effort parse of free prose, which cannot recover premise citations. Its
parse-failure rate is a reported result, not a hidden loss.

**Compute confound.** Verified conditions get one revision; others get none, so
they consume more samples. First-attempt and post-revision results are logged
separately, and the pre-registered comparisons keep compute matched: the
**structure** effect is tested at `struct_verify` *first attempt* vs `cot_fs`;
the **verification** effect is tested within-condition.

The verifier's critique names which steps failed and why, but never the answer.
A verifier that leaks the label is an oracle, and the study would be measuring
the oracle.

### 2.4 Datasets

| dataset | depth knob | gold steps | supports β_depth | role |
|---|---|---|---|---|
| **PrOntoQA** (fictional ontology) | exact, set by hop count | yes | **yes** | primary |
| **ProofWriter** OWA depth-5 | `QDep` 0–5, dataset-labelled | yes | **yes** | confirmatory |
| **FOLIO** validation | none | **no** | **no** | generalisation |
| **BBH** `logical_deduction` | object count (proxy) | no | no | OOD check |

PrOntoQA is primary because its fictional predicates (`zumpus`, `wumpus`) make
world knowledge useless — a correct answer must come through the premises, which
is the necessary condition for a propagation measurement to mean anything — and
because a synthetic ontology generated at run time from a fixed seed cannot be in
any pretraining corpus, neutralising contamination for the primary result.

**FOLIO ships no gold proof chain.** First-error position and recovery rate are
therefore structurally unavailable there and appear as blank cells, never as
estimates. Producing them would require a model to grade itself, which §2.1
forbids. This is a real limit on the generalisation claim and is treated as one.

### 2.5 Model, statistics, reproducibility

Subject model: **`gpt-4.1-mini-2025-04-14`** (changed from `claude-haiku-4-5-20251001`;
see `HYPOTHESIS.md` Amendment 1, recorded before any model call) — a dated
snapshot, so the run is reproducible, and mid-tier *on purpose*: a frontier model sits near ceiling on
PrOntoQA, and a ceiling gives β_depth ≈ 0 in every condition, i.e. no measurable
propagation. Headroom is a methodological requirement here, not a cost
compromise. `temperature=0.0`; seed varies item sampling and exemplar selection.

Statistics: ≥3 seeds; mean and 95% CI (Wilson for proportions) on every number;
every percentage carries its n; item-clustered paired bootstrap for Δβ (10,000
resamples); exact McNemar for paired accuracy; Holm–Bonferroni across the three
confirmatory comparisons. The logistic fit carries a small ridge penalty on the
slope, because depth-conditioned accuracy can separate perfectly in a small
sample and send the unpenalised MLE to infinity; the penalty shrinks toward zero
and is therefore conservative with respect to the hypothesis.

**Distinguishing "no effect" from "underpowered."** The minimum detectable Δβ at
80% power is computed by simulation at the realised n and realised per-depth
accuracies. A non-significant primary result is reported as *"no detected
effect, MDE = X"*, never as *"no effect"*. If the realised MDE exceeds the
pre-registered prediction of +0.15, the study is declared underpowered for its
own primary hypothesis, in the abstract.

Reproducibility: every dependency pinned exactly plus `uv.lock`; Python 3.12;
`make data` rebuilds every benchmark from public sources and writes
`checksums.sha256` (ProofWriter is verified against a pinned SHA-256 and the
pipeline refuses to proceed on mismatch); raw model outputs persisted to
`results/raw/`; `make reproduce` regenerates every table from those records
**with no network access**. The analysis pipeline is validated end-to-end against
synthetic records containing a planted +0.6 effect, confirming it recovers a
known effect and does not manufacture one under the null.

---

## 3. Results

The full per-condition tables, depth breakdowns, confirmatory tests and power
statement are generated directly from `results/raw/` by
`scripts/write_results.py` and live in **[results_full.md](results_full.md)**.
They are not transcribed by hand. What follows is the interpretation.

22,470 records · 24,234 API calls · 21.1M input + 4.0M output tokens · **$14.91**
· 13 calls failed after retries (0.05%) and are excluded from every denominator.

### 3.1 Only one of four benchmarks could measure the effect

The manipulation check — does chain-of-thought beat answering directly? — failed
on three of the four datasets, in two different ways:

| dataset | `direct_zs` | `cot_fs` | verdict |
|---|---:|---:|---|
| PrOntoQA | 98.9% | 98.9% | **fails (ceiling)** — no headroom |
| BBH `logical_deduction` | 97.9% | 98.7% | **fails (ceiling)** — no headroom |
| FOLIO | 76.8% | 76.3% | inconclusive; and no gold chains, so no β_depth |
| **ProofWriter** | 67.4% | 63.8% | **fails (inverted)** — CoT is 3.6 points *worse* |

PrOntoQA and BBH are saturated: a model that emits no reasoning at all scores
98–99%, so accuracy cannot decline with depth and β_depth ≈ 0 in every condition.
**C1, the pre-registered primary comparison, is therefore not estimable** — that
is what is reported, not a null.

ProofWriter is the only dataset with both headroom and gold depth labels, so it
carries the entire inferential weight of the study. That was pre-registered as
C2 (§2.3), not chosen after the fact — but a design meant to rest on two
instruments is resting on one.

### 3.2 Primary result: the intervention made propagation worse

**C2: Δβ = −0.174, 95% CI [−0.276, −0.077], p = 0.0008, Holm-adjusted
p = 0.0024, n = 865 paired items.** H1 predicted **+0.15**. The measured effect is
significant, and in the opposite direction: under `struct_verify` accuracy falls
*faster* with proof depth than under `cot_fs`.

This is a rejection of the pre-registered hypothesis, not a null. The study is
nonetheless **underpowered for the effect it predicted** — the realised MDE on
ProofWriter is +0.20, above the predicted +0.15 — so it could not have reliably
detected the hypothesised benefit had one existed. Both facts are reported
together because either alone would mislead.

### 3.3 The ablation says structure helped and verification hurt

The headline condition bundles two manipulations. Separating them (β_depth,
ProofWriter):

| condition | β_depth | accuracy | first attempt |
|---|---:|---:|---:|
| `cot_fs` (baseline) | −0.097 | 63.8% | 63.8% |
| `struct` (structure only) | **−0.007** | **70.2%** | 70.2% |
| `struct_verify` (structure + revision) | **−0.271** | 67.7% | **70.1%** |

Structure alone produces an almost perfectly flat depth curve — 74%, 64%, 70%,
71%, 74%, 66% across depths 0–5 — which is precisely H1's win condition. Adding
the verifier-triggered revision loop reverses it (82% → 51% across the same
range) and *lowers* accuracy: 70.1% on the first attempt falls to 67.7% after
revision.

A mechanism consistent with this: the verifier rejects 25% of emitted steps, and
a rejection triggers revision on chains that were frequently already correct. The
model then edits sound reasoning into unsound reasoning. Deeper problems have
more steps, so more opportunities for a spurious rejection, so more damaging
revisions — which is exactly how a revision loop would manufacture a depth
gradient.

**This decomposition is exploratory.** `Δβ(struct − cot_fs) = +0.090` is not one
of the three pre-registered comparisons and carries no confirmatory weight. It is
the obvious pre-registered follow-up, not a finding.

Consistent with the same picture, revision *helped* the unstructured condition
(`verify`: 67.0% → 70.7%) while hurting the structured one. Revision appears to
be useful exactly where the model had no reliable chain to damage.

### 3.4 Faithfulness moved as predicted, even as robustness got worse

H4 predicted post-error recovery would **fall** under the intervention, because
forcing checkable steps makes the chain load-bearing rather than decorative.
Comparing the two conditions where recovery is measured on the same footing:

- `struct` 78.6% → `struct_verify` **60.9%** (−17.7 points)

So the chain did become more causally connected to the answer, and errors
propagated harder as a result. **The intervention improved faithfulness while
degrading robustness** — the tension pre-registered in §2.1 as a possible
outcome, realised.

> **A measurement artefact that must not be read as a result.** The
> free-form conditions (`direct_*`, `cot_*`, `verify`) show a 100% verifier
> rejection rate and a mean first-error position of exactly 1.00. That is not a
> reasoning finding: prose does not cite premise ids, so every recovered step
> fails verification for want of a citation, at step 1. Recovery, rejection and
> first-error position are only comparable **among structured conditions**. Any
> comparison of those three columns between a structured and an unstructured
> condition is meaningless, and the `struct` vs `struct_verify` contrast above is
> the only one drawn.

### 3.5 The structured format is not free

On the two saturated datasets the structured format actively costs accuracy:
PrOntoQA `struct` 94.9% vs `cot_zs` 99.5%; BBH `struct` 88.2% vs `cot_fs` 98.7%.
BBH `logical_deduction` is an ordering puzzle, not a premise-citation task, so
being forced into typed triples is a poor fit. Structure helps where the task is
genuinely a derivation over citable premises and hurts where it is not.

### 3.6 H5 confirmed: unstructured output is not verifiable

`verify` applies the verifier to a best-effort parse of prose. On PrOntoQA the
parse-failure rate is **99.9% (868/869)** — far above the predicted >25%. Free-form
chain-of-thought essentially cannot be checked against the premises that are
supposed to license it. Whatever else this study shows, it shows that the
verifiability of a chain has to be designed in; it cannot be recovered after the
fact.

---

## 4. Threats to Validity

These are the study's own weakest points, named before any result exists so that
the list cannot be trimmed to fit a finding.

**The primary instrument saturated, and C1 could not be estimated.** This is the
single most important limitation of the study and it is not a subtle one. In the
50-item pilot, PrOntoQA `direct_zs` — answering with no chain of thought at all —
scored **98% (49/50)**, flat across depths 1–5 (11/11, 11/11, 11/11, 10/11, 6/6).
`cot_fs` scored the same 98%. With no accuracy gradient there is no slope:
β_depth ≈ 0 in every condition, so Δβ has nothing to vary against and **C1 is
reported as *not estimable*, never as a null**. §2.5's assumption that
`gpt-4.1-mini` is mid-tier *for this task* is simply false; 1–5 hop modus ponens
over a fictional ontology is trivial for it.

Both routes to a harder instrument were tested and both are closed: the PrOntoQA
generator hangs at 6, 8, 10 and 12 hops under the fictional ontology, and the
harder deduction rules (`ProofByContra`, `Composed`, `AndIntro`) require
`--proofs-only`, which emits proofs with no question–answer pairs and so cannot
produce the task. Relevant distractors, random premise ordering, and proof-width
2 all generate but yield structurally identical items. The consequence is that
**ProofWriter (C2) carries the primary inference** — a comparison pre-registered
in §2.3, not one invented after seeing data — and that a single dataset now
carries a claim the design intended two to share.

That saturation is itself a reportable observation: PrOntoQA at depths 1–5, the
instrument behind a widely-cited chain-of-thought analysis, no longer
discriminates between reasoning and no-reasoning for a mid-tier 2025 model.

**Contamination.** PrOntoQA's fictional ontology is generated at run time and
cannot be memorised, which protects the *primary* result. ProofWriter (2020),
FOLIO, and BBH are all public and plausibly in pretraining data; BBH in
particular is a widely-used eval. A direct-answer baseline above 95% is
automatically flagged in the output as possible leakage. Any confirmatory or
generalisation result carries this caveat in a way the primary result does not.

**Prompt sensitivity.** The structured condition asks for an unusual output
format. A measured difference could reflect format familiarity rather than
reasoning quality. Three things partially mitigate this and none fully resolve
it: all conditions see identical premises and question; exemplars are the
dataset's own gold proofs rendered into each condition's format, so no condition
gets a better *worked example*; and the `struct` ablation separates format from
feedback. A genuine resolution would require multiple surface realisations of the
same structural constraint, which this design does not include.

**Verifier coverage gaps.** The verifier decides a fragment: unary and binary
literals with universally-quantified implication rules. Within that fragment it
is exact and calibrated to 100% on 4,832 gold proofs. Outside it, it is silent —
negation-as-failure theories are excluded (~3.5% of ProofWriter), and FOLIO's
full first-order logic is not attempted at all. Verifier rejection rates are
therefore lower bounds on true error rates: a step the verifier cannot check is
not a step it certifies. Over-citation tolerance (bounded at 8 premises) is a
deliberate leniency and is reported separately.

**Depth is confounded with difficulty.** This is the most serious threat to the
primary metric, and it cannot be fully removed. A depth-5 item is not just a
depth-1 item with more steps: it usually has a larger premise set, more
distractors, and more opportunities for a wrong branch. β_depth therefore mixes
"errors propagate" with "deeper problems are harder for reasons unrelated to
chain length." Two things bound the damage — PrOntoQA's generator holds ontology
size and distractor policy fixed while varying only hop count, and the metric of
interest is a *difference* of slopes between conditions on identical items, so
any difficulty confound common to both conditions cancels. What does not cancel
is an *interaction*: if the intervention helps more on small premise sets, that
would appear as Δβ > 0 without any change in propagation. Disentangling this
would need premise-set size varied orthogonally to depth, which this design does
not do.

**Single model family.** One model, one provider (OpenAI), one snapshot. Nothing
here licenses a claim about language models in general. The harness is
provider-agnostic and a second family can be run to address this directly; that
is the single highest-value extension. The mid-tier choice is
methodologically motivated (headroom), but it also means the result may not
transfer to frontier models, which may fail differently — or not enough to
measure.

**Answer-space asymmetry.** PrOntoQA is binary (True/False), so a 50% baseline is
available by guessing; ProofWriter and FOLIO are three-way. Accuracy is therefore
not comparable *across* datasets, only across conditions within one.

**Revision confound.** Verified conditions issue a second call. First-attempt
results are logged separately and the pre-registered comparison is compute-
matched, but a residual concern remains: the revision prompt shows the model its
own prior attempt, which is a different intervention from simply sampling twice.
A no-op-critique control would isolate this and is not included.

**Bootstrap resolution.** The bootstrap cannot resolve p below 1/B; reported
p-values are floored at 1/10,000 rather than shown as zero.

**A scoring bug was found and fixed after the run, from raw records.** BBH's
gold targets are written `(A)` while the model answers `A`; compared as raw
strings this scored every zero-shot BBH item wrong and produced apparent
accuracies of 0.0% for `direct_zs` and `cot_zs`. It was caught because 0.0%
alongside 66.2% for `direct_fs` is not a plausible model behaviour. Because raw
response text is persisted, the correction re-derived every affected number with
no new API calls. That this was recoverable is a property of the design; that it
existed at all is a reminder that a scoring layer is as capable of producing a
confident wrong answer as a model is.

**One instrument carries the study.** The design intended PrOntoQA and
ProofWriter to test the same hypothesis independently. Saturation removed
PrOntoQA, and FOLIO and BBH cannot support β_depth at all. A single benchmark,
one model, one snapshot now carries the entire confirmatory claim. The negative
result should be read as *this intervention, on this benchmark, with this
model* — not as a general claim about structured reasoning.

---

## 5. Limitations

Beyond the threats above:

- **Unknown-answer items carry no step metrics.** Under the open-world
  assumption, ProofWriter's `Unknown` items have no positive proof by
  construction. They contribute to accuracy but not to first-error position or
  recovery, so step-level denominators are smaller than accuracy denominators.
  Every table states both.
- **`verify` is a deliberately weak ablation.** Applying a citation-checking
  verifier to free prose will fail often. That is the measurement, but it means
  `verify` is not a fair standalone method — only a probe of what structure buys.
- **Exemplar selection is depth-spread, not matched.** Few-shot exemplars span
  depths but are not matched to the target item's depth.
- **No human validation of the verifier's error taxonomy.** Verdict categories
  are checked against gold proofs, not against human judgements of what a
  reasoning error is.
- **One benchmark carries the primary claim.** ProofWriter is confirmatory, but if
  PrOntoQA's synthetic register is unrepresentative, the primary result is
  narrow. FOLIO would be the natural corrective and cannot serve, because it has
  no gold chain.

---

## 6. Reproduction

```bash
make setup && make data && make test
make auth-check     # verify credentials, confirm the model (1 tiny call)
make pilot          # 50 items/condition, ~$0.76  (needs OPENAI_API_KEY)
make full           # full grid, ~$24.68          (needs OPENAI_API_KEY)
make reproduce      # regenerate every table, NO network
```

`make reproduce` reads only `results/raw/` and needs no API key. Pre-registration
is `HYPOTHESIS.md`, committed at `4325878` before any model call.

---

## References

- Saparov & He (2023). *Language Models Are Greedy Reasoners: A Systematic Formal
  Analysis of Chain-of-Thought.* ICLR. [arXiv:2210.01240](https://arxiv.org/abs/2210.01240)
- Saparov et al. (2023). *Testing the General Deductive Reasoning Capacity of
  Large Language Models Using OOD Examples.* NeurIPS.
- Tafjord, Dalvi & Clark (2021). *ProofWriter: Generating Implications, Proofs,
  and Abductive Statements over Natural Language.* Findings of ACL.
- Han et al. (2022). *FOLIO: Natural Language Reasoning with First-Order Logic.*
  [arXiv:2209.00840](https://arxiv.org/abs/2209.00840)
- Suzgun et al. (2022). *Challenging BIG-Bench Tasks and Whether Chain-of-Thought
  Can Solve Them.* [arXiv:2210.09261](https://arxiv.org/abs/2210.09261)

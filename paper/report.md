# Where Reasoning Breaks: Measuring Error Propagation in Multi-Step LLM Deduction

*Independent research project. Not affiliated with, endorsed by, or conducted under any research lab.*

---

## Abstract

> **STATUS: NO RESULTS YET. The experiment has not been run — it requires an API
> key that was not available in the build environment. Every number below is a
> placeholder marked `[PENDING]`. Nothing in this document should be read,
> cited, or quoted as a finding.**
>
> **When the run completes, this abstract will state the direction of the
> primary result in its first two sentences, in bold, including the case where
> the intervention did not work.**

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

> **`[PENDING]` — the experiment has not been run.**
>
> The apparatus is complete and validated; execution requires an `ANTHROPIC_API_KEY`.
> The 50-item pilot projects at **$0.76** and the full run (7 conditions × 4
> datasets × 3 seeds × 290 items = 22,470 calls + up to 6,960 revisions) projects
> at **$24.68**, inside a hard $25 ceiling. n was reduced 300 → 290 to fit that
> ceiling while keeping every dataset, condition, and seed; see Amendment 2.
>
> This section will contain, with no omissions: the per-condition table for all
> seven conditions on all four datasets; accuracy by proof depth; the three
> pre-registered confirmatory tests with raw and Holm-adjusted p-values; the
> power/MDE statement; contamination flags; and a run-health count of API
> failures and unparseable responses.

### 3.1 Manipulation check `[PENDING]`
### 3.2 Primary result: depth-conditioned slope `[PENDING]`
### 3.3 Where chains break: first-error position `[PENDING]`
### 3.4 Faithfulness: post-error recovery `[PENDING]`
### 3.5 Ablations: structure vs verification `[PENDING]`
### 3.6 Generalisation: FOLIO and BBH `[PENDING]`

---

## 4. Threats to Validity

These are the study's own weakest points, named before any result exists so that
the list cannot be trimmed to fit a finding.

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

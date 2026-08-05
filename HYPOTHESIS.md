# Pre-Registration

**Frozen: 2026-08-05, before any model call was made.**
Nothing below may change after the first API call. Any metric introduced later is
exploratory and is labelled as such in the report. Git history is the audit trail:
this file's first commit precedes every commit that touches `results/`.

---

## 1. Research question

Does forcing a language model to commit to explicit, machine-checkable intermediate
steps improve multi-step logical reasoning accuracy, and does it reduce **error
propagation** — the tendency for one incorrect intermediate step to corrupt every
step after it?

The first half is established (chain-of-thought helps) and is included only as a
manipulation check. The contribution is the second half: measuring *where in a chain*
reasoning breaks, and whether structural constraints change the shape of that failure
curve.

## 2. Primary metric (ONE, pre-committed)

**Δβ = β_depth(struct_verify) − β_depth(cot_fs)** on **ProntoQA, fictional ontology.**

β_depth is the coefficient on ground-truth proof depth in a per-condition logistic
regression fit over items:

```
logit( P(final answer correct) ) = α + β_depth · depth
```

β_depth is necessarily negative (accuracy falls with depth). It **is** the propagation
number: a flatter (less negative) slope means an error at depth *d* costs less at
depths > *d*. Δβ > 0 is the win condition.

**Primary test:** two-sided paired bootstrap over items (10,000 resamples, clustered
by item so the same item is resampled in both conditions), H₀: Δβ = 0. Significance
α = 0.05 after Holm–Bonferroni correction across the three confirmatory comparisons
in §4.

**Why ProntoQA is primary, not ProofWriter or FOLIO.** ProntoQA's fictional ontology
(`zumpus`, `wumpus`) makes world knowledge useless, so a correct answer must come
through the premises — the necessary condition for a propagation measurement to mean
anything. Hop count is set by me exactly, not inferred. And a synthetic ontology
generated at run time from a fixed seed cannot be in any pretraining corpus, which
neutralises contamination for the primary result. ProofWriter is confirmatory
(§4). FOLIO and BBH are generalisation checks and **cannot** support the primary
metric — see §6.

## 3. Secondary metrics (pre-committed, reported for every condition)

| Metric | Definition | Denominator |
|---|---|---|
| **Answer accuracy** | exact match on final label after normalisation | all items |
| **First-error position** | 1-based index of earliest step whose derived proposition is not entailed by its cited premises under the dataset's own annotations; `None` if no incorrect step | items with ≥1 incorrect step |
| **Post-error recovery rate** | fraction reaching the correct final answer *despite* containing ≥1 incorrect step | items with ≥1 incorrect step |
| **Verifier rejection rate** | fraction of emitted steps the non-model verifier rejects | all emitted steps |
| **Parse-failure rate** | fraction of responses from which no derivation could be extracted | all items |

Step-level correctness is scored **only** against dataset proof annotations
(ProntoQA `chain_of_thought`; ProofWriter `proofsWithIntermediates` + s-expression
`representation`). No model ever grades an unlabelled chain. Where a dataset lacks
step annotations, first-error position and recovery rate are **not reported** — they
are left blank, not estimated.

### Direction of the recovery-rate prediction

Post-error recovery is **not** a "higher is better" metric, and this is the crux of
the explainability claim. High recovery means the model reached the right answer
while its own stated chain was broken — i.e. the chain was decorative, not causal.

Prediction: `struct_verify` **lowers** recovery relative to `cot_fs`, because forcing
checkable steps makes the chain more load-bearing. Errors that do occur should
propagate *harder*, while occurring *less often*. Δβ and recovery can therefore move
in opposite directions, and **that combination is a positive result for faithfulness,
not a contradiction.** Pre-registering this now so it cannot be reframed later.

## 4. Conditions and confirmatory comparisons

Seven conditions. `k=4` few-shot exemplars throughout.

| id | structure | verifier | description |
|---|---|---|---|
| `direct_zs` | – | – | answer only, zero-shot |
| `direct_fs` | – | – | answer only, few-shot |
| `cot_zs` | – | – | free-form CoT, zero-shot |
| `cot_fs` | – | – | free-form CoT, few-shot with gold chains |
| `struct` | ✓ | – | typed triples, no verifier feedback |
| `verify` | – | ✓ | free-form CoT, best-effort parse, verifier feedback |
| `struct_verify` | ✓ | ✓ | typed triples + verifier feedback |

A **typed triple** is `(premise_ids, rule_id, derived_proposition)`. A Python verifier
with no model in it resolves the cited ids against the premise set, checks that the
cited rule actually fires on those premises, and checks that the stated conclusion is
what the rule yields. A step citing premises that do not entail it is a *detectable*
error, not a stylistic one.

`verify` (verification without structure) applies the same verifier to a best-effort
parse of free-form CoT; when parsing fails, no feedback is given. The parse-failure
rate is itself a reported result — it measures how much of any benefit comes from
being *verifiable at all* rather than from the checker.

**Compute confound and how it is controlled.** `verify` and `struct_verify` get
exactly one verifier-triggered revision; other conditions get none, so they consume
fewer samples. To keep the structure effect uncontaminated by the extra sample, every
verified condition logs **first-attempt** and **post-revision** results separately, and:

- the **structure** effect is tested at matched compute: `struct_verify` *first attempt*
  vs `cot_fs`;
- the **verification** effect is tested within-condition: `struct_verify` post-revision
  vs `struct_verify` first attempt.

Three confirmatory comparisons carry the Holm–Bonferroni family:

1. **C1 (primary)** Δβ, `struct_verify` vs `cot_fs`, ProntoQA.
2. **C2** Δβ, `struct_verify` vs `cot_fs`, ProofWriter OWA depth-5.
3. **C3** answer accuracy, `struct_verify` vs `cot_fs`, ProntoQA (McNemar, exact, paired on items).

Everything else — all 7 conditions × all metrics × FOLIO × BBH — is descriptive and
reported without inferential claims.

## 5. Predictions

- **H1 (primary):** Δβ > 0 on ProntoQA. Predicted effect small: **Δβ ≈ +0.15 logits/hop**.
- **H2:** answer accuracy rises for `struct_verify` over `cot_fs`, predicted **+4 to +10 points**.
- **H3:** first-error position moves *later* under `struct_verify`.
- **H4:** post-error recovery rate **falls** under `struct_verify` (see §3).
- **H5:** `struct` alone captures most of the effect; `verify` alone captures little,
  because unstructured output is often unparseable. Predicted parse-failure rate for
  `verify` > 25%.

**Stated plainly in advance: I expect H1 to be the hardest to detect and consider a
null result on it the most likely single outcome.** The apparatus is built to
distinguish "no effect" from "underpowered", and §7 defines how.

## 6. Datasets and what each can support

| dataset | depth knob | gold steps | supports β_depth | role |
|---|---|---|---|---|
| **ProntoQA** (fictional) | exact, set by `--min-hops/--max-hops` | yes (`chain_of_thought`) | **yes** | primary |
| **ProofWriter** OWA depth-5 | `QDep` 0–5, dataset-labelled | yes (`proofsWithIntermediates`) | **yes** | confirmatory |
| **FOLIO** validation | none | **no** | **no** | generalisation, answer accuracy only |
| **BBH** logical_deduction | object count (3/5/7) as proxy | no | no | OOD check only |

FOLIO has no gold proof chain, so first-error position and recovery rate are
structurally unavailable there and will be blank in every table. Reporting them would
require a model to grade itself, which §3 forbids.

Known data quirks, handled explicitly in the loader:
- FOLIO labels its third class `Unknown` in train but `Uncertain` in validation, and
  the two splits do not share columns (train lacks `conclusion-FOL`; validation lacks
  `example_id`/`story_id`). Normalised to `{True, False, Unknown}`.
- ProofWriter answers are mixed-type: Python `True`/`False` booleans and the *string*
  `"Unknown"`. Normalised.
- ProofWriter "depth-5" contains a 10-item tail at QDep 6–7 out of 20,030. Excluded
  from slope fits; the exclusion is logged.

## 7. Statistics, power, and the null/underpowered distinction

- **≥3 seeds per condition.** Seed controls item sampling, few-shot exemplar choice,
  and ProntoQA ontology generation. Recorded per run.
- Mean and **95% CI** on every number. **Every reported percentage carries its n.**
- Paired tests on identical items: bootstrap for β, exact McNemar for accuracy.
- **Holm–Bonferroni** across C1–C3.
- **Target n:** 300 items per condition per dataset per seed (50 per depth bin × 6
  bins), 3 seeds.
- **Power:** the minimum detectable effect for C1 is computed by simulation at the
  realised n and realised per-depth accuracies, and reported in the paper. A
  non-significant C1 is reported as **"no detected effect, MDE = X"**, never as
  "no effect". If the realised MDE exceeds the H1 prediction of +0.15, the study is
  declared **underpowered for its own primary hypothesis** and says so in the abstract.

## 8. Model under test

**`claude-haiku-4-5-20251001`** — recorded in every result file.

Chosen for two reasons, both pre-committed:
1. It is a **dated snapshot**, so the run is reproducible. Undated aliases move.
2. It is **mid-tier on purpose.** A frontier model sits near ceiling on ProntoQA, and
   a ceiling gives β_depth ≈ 0 in every condition — no slope, no measurable
   propagation, no result. Headroom is a methodological requirement here, not a cost
   compromise.

Sampling: `temperature=0.0` for the main runs; seed varies item sampling and exemplar
selection, not decoding. Single model family is a named threat to validity (§ report).

## 9. Stopping rule

The full run executes exactly the design above and stops. No peeking-and-extending: if
C1 is non-significant, that is the result. No condition is dropped for being
unflattering — all seven are reported, including failures and crashes.

## 10. Honesty commitments

- If the intervention does not help, the abstract says so **in bold, in the first two
  sentences.**
- Every condition run is reported. No quiet dropping.
- A suspiciously high baseline is flagged as possible contamination in the report.
- No number appears that cannot be regenerated from `results/raw/` by `make reproduce`
  with no network access.

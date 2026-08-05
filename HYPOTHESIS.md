# Pre-Registration

**Frozen: 2026-08-05, before any model call was made.**
Nothing below may change after the first API call. Any metric introduced later is
exploratory and is labelled as such in the report. Git history is the audit trail:
this file's first commit precedes every commit that touches `results/`.

---

## Amendments

Recorded openly rather than edited in place. Original text below is left
untouched so the diff is auditable. **No amendment may touch the primary metric
(§2), the confirmatory comparisons (§4), or the predictions (§5)** — those are
what pre-registration exists to protect, and they are unchanged.

### Amendment 1 — 2026-08-05, before any model call

**Subject model changed** from `claude-haiku-4-5-20251001` (§8) to
**`gpt-4.1-mini-2025-04-14`**.

*Reason:* resource availability. No Anthropic API credential was obtainable for
this project; an OpenAI credential was. The only alternative Anthropic route was
a Claude Code subscription OAuth token, which was rejected on two grounds: it is
a subscription credential not provisioned for batch inference, and routing the
study through Claude Code would inject an agent system prompt and tools and would
not honour `temperature=0.0`, breaking the decoding config pinned in §8.

*Why this does not compromise the pre-registration:* the model was always a
resource choice, not a hypothesis. §8's stated selection criteria are unchanged
and the replacement satisfies both — it is a **dated snapshot** (reproducible,
unlike a moving alias) and it is **mid-tier**, preserving the headroom that a
depth-slope measurement requires. The study tests a prompting intervention, not a
vendor.

*Consequence for claims:* the "single model family" threat in the report now
reads **OpenAI**, not Anthropic. Nothing here licenses a claim about language
models in general, and the harness is now provider-agnostic specifically so a
second family can be added later to address that threat directly.

**Still true and still binding:** `temperature=0.0`; the model id is recorded in
every result record; seeds vary item sampling and exemplar selection only.

### Amendment 2 — 2026-08-05, before any model call

**Target n reduced from 300 to 290** items per condition per dataset per seed
(§7), to fit a hard $25 budget ceiling. Projected spend at n=290 is $24.68.

*Reason:* budget, stated in advance. The alternative was dropping FOLIO and BBH
entirely; a 3.3% reduction in n was preferred because it preserves **all four
datasets, all seven conditions, and all three seeds**, and therefore both
confirmatory slope tests (C1, C2) and the accuracy test (C3) exactly as
pre-registered. Dropping datasets would have removed the generalisation and OOD
checks outright.

*Consequence for claims:* a marginal power loss. The realised MDE is computed
from the realised n regardless (§7), so this cannot silently turn an
underpowered study into an apparently null one — if 290 is too few, the report
says so in the abstract.

### Amendment 3 — 2026-08-05, after the pilot, before the full run

**C1 (the primary comparison) is expected to be NOT ESTIMABLE, because ProntoQA
is at ceiling for this model.** Recorded here in advance of the full run so the
outcome cannot be presented later as anything other than what it is.

*Evidence, from the 50-item pilot (`results/raw/pilot/`, separate phase, never
pooled into the confirmatory analysis):*

| ProntoQA condition | accuracy | by depth |
|---|---|---|
| `direct_zs` (no reasoning at all) | 98% (49/50) | d1 11/11 · d2 11/11 · d3 11/11 · d4 10/11 · d5 6/6 |
| `cot_fs` | 98% (49/50) | d1 11/11 · d2 11/11 · d3 11/11 · d4 10/11 · d5 6/6 |
| `struct_verify` | 96% (48/50) | d1 11/11 · d2 11/11 · d3 10/11 · d4 10/11 · d5 6/6 |

*What triggered this amendment:* the **manipulation check failed**. Direct answer
without any chain scores 98%, so chain-of-thought has no room to help and
accuracy is flat across depth. β_depth ≈ 0 in every condition, so Δβ has no
denominator to vary against. **Δβ itself was not computed before writing this** —
the trigger is a ceiling in raw accuracy, not a look at the primary effect.

*This was predicted.* §8 states that a model near ceiling on ProntoQA "gives
β_depth ≈ 0 in every condition — no slope, no measurable propagation, no
result." The §8 assumption that `gpt-4.1-mini` is mid-tier **for this task** is
falsified: 1–5 hop modus ponens over a fictional ontology is trivial for it.

*Why the instrument cannot simply be made harder.* Both routes were tested and
both are closed:
- **Deeper chains:** the ProntoQA generator hangs at 6, 8, 10, and 12 hops with
  the fictional ontology (consistent with its own "insufficient concept names"
  warnings). 5 hops is the practical ceiling.
- **Harder deduction rules:** `ProofByContra`, `Composed`, and `AndIntro` all
  require `--proofs-only`, which emits proofs without question–answer examples
  and so cannot produce the task at all. Relevant distractors, random premise
  ordering, and proof-width 2 generate successfully but yield structurally
  identical items (chain length 11, 25 premises).

*Consequence, and what does NOT change:*
- **C1 will be reported as "not estimable — ceiling effect", in the abstract.**
  It is not reported as a null result, and not quietly dropped.
- **C2 (ProofWriter) carries the primary inference.** C2 was pre-registered in §4
  as a confirmatory comparison; it is not a new test invented after seeing data.
  The pilot shows ProofWriter has a real gradient (`cot_fs`: d0 8/9 → d4 5/9 →
  d5 2/4), so the slope is estimable there.
- The primary **metric** (Δβ, §2), the comparison set (§4), and the predictions
  (§5) are unchanged. Only the estimability of one of the three is now known.
- ProntoQA is still run in full and still reported. Its ceiling is a result about
  the benchmark, not a reason to hide it: a benchmark that anchored a widely-cited
  chain-of-thought analysis is saturated at these depths for a mid-tier 2025 model.

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
  bins), 3 seeds. *(Amended to 290 by Amendment 2 — budget ceiling.)*
- **Power:** the minimum detectable effect for C1 is computed by simulation at the
  realised n and realised per-depth accuracies, and reported in the paper. A
  non-significant C1 is reported as **"no detected effect, MDE = X"**, never as
  "no effect". If the realised MDE exceeds the H1 prediction of +0.15, the study is
  declared **underpowered for its own primary hypothesis** and says so in the abstract.

## 8. Model under test

**`claude-haiku-4-5-20251001`** — recorded in every result file.

> **SUPERSEDED by Amendment 1** (top of file): the subject model is now
> `gpt-4.1-mini-2025-04-14`. The two selection criteria stated below are
> unchanged and the replacement satisfies both. Original text kept verbatim.

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

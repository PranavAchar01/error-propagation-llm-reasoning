# Results — phase `synthetic`

Model: `synthetic-model-v0`  ·  raw records: 1,200  ·  seeds: [1]

Every percentage carries its denominator. `--` means the quantity is not
measurable for that dataset (FOLIO and BBH ship no gold proof chain), not zero.

## Per-condition summary

| dataset | condition | n | accuracy [95% CI] | first-attempt acc | beta_depth | first-error pos | post-error recovery | verifier reject | parse fail |
|---|---|---|---|---|---|---|---|---|---|
| prontoqa | `cot_fs` | 600 | 56.5% [52.5%, 60.4%] | 56.5% | -0.861 | 2.00 | 0.0% (n=261) | 14.5% (n=1800) | 0.0% (n=600) |
| prontoqa | `struct_verify` | 600 | 89.0% [86.2%, 91.3%] | 89.0% | -0.308 | 3.00 | 0.0% (n=66) | 3.7% (n=1800) | 0.0% (n=600) |

## Accuracy by ground-truth proof depth

### prontoqa

| condition | d=1 | d=2 | d=3 | d=4 | d=5 | beta |
|---|---|---|---|---|---|---|
| `cot_fs` | 92% (111/120) | 75% (90/120) | 52% (62/120) | 43% (52/120) | 20% (24/120) | -0.861 |
| `struct_verify` | 93% (112/120) | 92% (111/120) | 92% (110/120) | 85% (102/120) | 82% (99/120) | -0.308 |

## Confirmatory tests (pre-registered)

- **C1** (prontoqa, delta_beta `struct_verify` - `cot_fs`): +0.553 [95% CI +0.285, +0.803], p=0.0033, n=600, 300 resamples  ·  Holm-adjusted p=0.0033 -> **REJECT H0**

- **C2** (proofwriter, `struct_verify` vs `cot_fs`, delta_beta): no paired items — not run.

- **C3** (prontoqa, accuracy `struct_verify` - `cot_fs`): +32.5 points, exact McNemar b=229 c=34, p=0.0000, n=600  ·  Holm-adjusted p=0.0000 -> **REJECT H0**

## Power

- **prontoqa**: n=600 paired items. Minimum detectable delta_beta at 80% power = **+0.35** logits/depth. Therefore the study is **underpowered for its own primary hypothesis** (MDE exceeds the pre-registered prediction of +0.15).

- proofwriter: too few paired items (0) to estimate power.
## Contamination checks

- No direct-answer baseline exceeded 95%.

## Run health

- API calls that failed after retries and were excluded: **0**
- Responses with no parseable ANSWER line (scored incorrect): **0**


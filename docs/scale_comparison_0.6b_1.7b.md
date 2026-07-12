# 0.6B vs 1.7B scale comparison (Qwen3-Base)

Status 2026-07-12: P2.5 / P3 / P4 complete and registered at both scales.
**P5.1 at 1.7B is blocked on the Modal workspace spend limit.** Six launch
attempts failed across two distinct causes: genuine L4/A100 worker preemptions
(fixed by adding server-side `retries` to the Modal function) and `.remote()`
input cancellations on local-client disconnect (fixed by adding a `--spawn`
launch mode). With both fixes in place, the final attempt surfaced the real
blocker explicitly: `App creation failed: workspace billing cycle spend limit
reached`. This is a configured per-cycle cap (started at $13), separate from
the account balance; the earlier empty-reason cancellations were most likely
the same cap. No partial artifacts were written.

Unblock: raise the workspace billing-cycle spend limit in the Modal dashboard
(Settings → Usage & Billing), then relaunch on **L4** (proven to complete a
5.1 run, cheaper and better capacity than A100), now hardened with spawn +
retries:

```bash
modal run --detach cloud/modal_run.py --prototype 51 --config configs/prototype51_qwen3_1.7b.yaml --gpu L4 --timeout-minutes 300 --spawn
# then, once the artifacts land on the volume:
python cloud/fetch_run.py --prototype 51 --run-id prototype51-qwen3-1-7b --register
```

Method: every 1.7B config is a minimal diff of its accepted 0.6B
pre-registration (experiment label, bundle paths, extraction model name; P5.1
additionally pre-registers the length-normalized scoring mode). Config tests
enforce the minimal-diff rule, so scale differences are attributable to the
model. Same seed (42), story generator, prompts, strengths, and gates.
Registered run IDs are in `results/index.jsonl` (`*-qwen3-1-7b` experiments).

Interpretation caveat, as everywhere in this project: all findings are
functional/causal observations about model internals and behavior, not
evidence of subjective experience.

## Prototype 2.5 — extraction and semantic validation

| Metric | 0.6B | 1.7B |
| --- | --- | --- |
| Nested P1 hard gates | pass | pass |
| Held-out accuracy | 0.6125 | **0.8125** |
| Held-out macro AUC | 0.9038 | **0.9535** |
| Mean correct margin | 3.81 | 27.68 |
| Selected layer | 19 | 19 |
| Validation soft gates | pass | **fail** |
| Shuffled-label macro AUC | 0.515 | 0.628 |
| Lexical robustness accuracy (chance 0.25) | 0.50 | 0.25 |
| Intensity mean Spearman | 0.125 | 0.0 |
| Decision-rule recommendation | proceed to geometry | revise before geometry |

The most interesting scale result so far: the 1.7B model separates held-out
emotion stories much better, but the vectors are **less semantically robust**
— lexical paraphrase control at chance, no intensity monotonicity, and more
signal surviving label shuffling. Better classification does not imply better
emotion-concept vectors under this extraction recipe. Nothing was tuned in
response; the downstream 1.7B runs carry this caveat and remain interpretable
as causal assays of these particular vectors.

## Prototype 3 — geometry

| Metric | 0.6B | 1.7B |
| --- | --- | --- |
| Valence Spearman | 0.0 | 0.0 |
| Arousal Spearman | 0.086 | 0.143 |
| Mean abs off-diagonal cosine | 0.330 | 0.315 |

The weak valence/arousal geometry caveat is **scale-stable**: four compact
vectors, near-orthogonal, no valence alignment at either size.

## Prototype 4 — causal steering

| Metric | 0.6B | 1.7B |
| --- | --- | --- |
| Hard gates | pass | pass |
| Soft gates | pass | pass |
| Dose-response Spearman | 1.0 | 1.0 |
| Target delta (positive strengths) | +0.065, CI [0.044, 0.092] | +0.014, CI [0.006, 0.019] |
| Specificity delta | +0.091, CI [0.070, 0.125] | +0.020, CI [0.015, 0.024] |
| Specificity − target | +0.026, CI [0.009, 0.035], p=0.25 | +0.006, CI [−0.001, +0.011], p=0.25 |
| Free-gen lexical specificity | 0.25 | 0.063 |

(CIs are prompt-cluster bootstrap; p-values are exact cluster sign-flip tests;
see docs/statistics.md. 0.6B CIs computed post hoc with `fe-analyze-stats`;
1.7B CIs are embedded in the run's metrics natively.)

Steering remains causally effective and monotone at 1.7B, but the same raw
strengths (−4…+4, unscaled) move a 2048-dim residual stream relatively less
than a 1024-dim one, so absolute deltas shrink. Cross-scale magnitude
comparisons need norm-relative strengths (a candidate methods change for a
future pre-registration, not retrofitted here).

**Specificity adjudication resolved across scales:** the 0.6B observation that
specificity delta exceeded target delta was suggestive (CI excluded zero) but
underpowered (exact p = 0.25 with 4 prompt clusters). At 1.7B the excess does
not replicate (CI includes zero). Verdict: **consistent with noise; not an
established effect.**

## Prototype 5.1 — robust activity preferences

Pending the 1.7B run (see status above). The 0.6B reference: mean expected
effect +0.372 aggregate / +0.354 row-level, pair-cluster CI [0.12, 0.59],
real−random paired contrast +0.342 (exact p = 0.047), best diagnostic layer
16. The 1.7B run will additionally report the new
`option_text_mean_logprob_margin` diagnostic and native statistics.

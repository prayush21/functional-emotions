# 0.6B vs 1.7B scale comparison (Qwen3-Base)

Status 2026-07-31: **P2.5 / P3 / P4 / P5.1 complete and registered at both
scales.** The ladder is now fully replicated at 1.7B.

P5.1 at 1.7B was blocked for roughly three weeks on the Modal workspace
billing-cycle spend limit (a configured per-cycle cap, separate from account
balance), which surfaced only after two unrelated failure modes were fixed:
genuine L4/A100 worker preemptions (fixed by server-side `retries` on the
Modal function) and `.remote()` input cancellations on local-client
disconnect (fixed by the `--spawn` launch mode). No partial artifacts were
written during any failed attempt. Raising the cap unblocked it; the run then
completed on **L4** on the first attempt, with both hardening fixes in place:

```bash
modal run --detach cloud/modal_run.py --prototype 51 --config configs/prototype51_qwen3_1.7b.yaml --gpu L4 --timeout-minutes 300 --spawn
python cloud/fetch_run.py --prototype 51 --run-id 20260730T065411Z__prototype51-qwen3-1-7b__qwen-qwen3-1-7b-base__seed-42__39cd15c820 --register
```

Operational note for future scale-ups: use **L4, not A100** (A100 was
capacity-starved and sat hours in the schedule queue; L4 completes a 5.1 run
comfortably and 1.7B float32 fits its 24GB).

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

| Metric | 0.6B | 1.7B |
| --- | --- | --- |
| Hard gates | pass | pass |
| Soft gates (9/9) | pass | pass |
| Mean expected effect (aggregate) | +0.372 | +0.101 |
| Expected effect, row-level | +0.354 | +0.099 |
| Pair-cluster CI | [0.120, 0.590] | [0.068, 0.137] |
| Sign-flip p (by pair) | 0.055 | **0.0078** |
| real − random | +0.342, p = 0.047 | +0.102, p = **0.0078** |
| real − wrong-emotion | +0.486, p = 0.094 | +0.139, p = **0.0078** |
| Dose-response Spearman | 1.0 | 1.0 |
| Best diagnostic layer | 16 | 16 |
| Per-emotion CIs excluding zero | 3 / 4 (`afraid` spans zero) | **4 / 4** |

**The Prototype 5 null does not survive at either scale, and the 1.7B result
is statistically stronger despite a smaller effect.** Magnitude attenuates
~3.5× (same residual-stream dilution seen in P4, see above), but every
inferential statistic tightens: the exact sign-flip test moves from marginal
(p = 0.055) to clearly significant (p = 0.0078), both control contrasts reach
the exact-test floor, and all four per-emotion CIs exclude zero where at 0.6B
`afraid` did not. Larger models give a smaller but far more *consistent*
effect across prompt-pair clusters, which is what the cluster-level tests
reward. Cross-scale magnitude comparison still needs norm-relative strengths.

**Mechanism of the original null, confirmed across scales.** The per-scoring-
mode breakdown (real vectors, positive strengths, row-weighted) isolates it:

| Scoring mode | 0.6B | 1.7B |
| --- | --- | --- |
| `choice_token_margin` (what P5 used) | −0.016 | −0.002 |
| `option_text_logprob_margin` (P5.1 primary) | +0.372 | +0.101 |
| `option_text_mean_logprob_margin` (length-normalized) | n/a | +0.029 |

Single-token A/B choice scoring is **null at both scales**, while option-text
scoring recovers a clear effect at both. This is direct evidence that the
Prototype 5 null was a measurement artifact of brittle choice-token scoring
rather than an absence of causal preference structure — and the diagnosis is
scale-stable, not a 0.6B accident. The length-normalized mode is positive but
attenuated, so the summed mode's effect is partly, but not wholly, carried by
option length; summed remains primary as pre-registered.

Standing caveat unchanged: these are compact four-emotion vectors with weak
valence/arousal geometry (P3, scale-stable). P5.1 tests assay robustness, not
mature emotion structure.

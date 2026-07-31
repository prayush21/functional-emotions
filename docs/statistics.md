# Statistics layer

Added after Prototype 5.1 (2026-07-10) as **additional reporting**. Nothing in
this layer is a pre-registered hard gate, and none of the completed prototypes'
gates were modified. Interpretation caveat, as everywhere in this project:
results are functional/causal findings about model internals and behavior, not
evidence of subjective experience.

## Components

- `src/functional_emotions/stats.py` — generic primitives:
  - `bootstrap_mean_ci` — percentile bootstrap CI for a mean, resampling whole
    clusters (story/pair/prompt level) so correlated rows stay together.
  - `sign_flip_pvalue` — two-sided cluster sign-flip permutation test of
    mean == 0; enumerates all flips exactly when the cluster count is small.
  - `paired_difference_stats` — CI plus sign-flip p for paired differences.
  - `multi_seed_summary` — combines one estimate per seed (3-5 seeds) with a
    Student-t interval, range, and sign agreement.
- `src/functional_emotions/analysis.py` — prototype-aware reporting:
  - `fe-analyze-stats --run-dir <bundle> [--run-dir ...] [--write]` computes
    the report for Prototype 4 and 5.1 bundles from their stored per-row
    diagnostics; multiple `--run-dir` values of the same prototype produce an
    across-seed summary. `--write` saves `stats_summary.json` into the bundle.
  - `prototype4.py` and `prototype51.py` call the same helpers so future runs
    embed a `statistics` block in `metrics.json` and CI fields in
    `summary.json`. An optional `statistics:` config block overrides
    `n_resamples` / `n_permutations` / `seed`.

Cluster levels: Prototype 4 clusters on `prompt_id`; Prototype 5.1 clusters on
`base_pair_id` (primary) with a `context_id` clustering as a sensitivity check.

## Findings on the registered 0.6B runs

### Prototype 4 specificity question (run `20260630T193015Z__prototype4__...__7d7df21e6d`)

Question: the specificity delta (0.091) exceeded the target delta (0.065) —
real or noise? On real positive-strength rows (12 rows, 4 prompt clusters):

- target delta +0.065, 95% bootstrap CI [0.044, 0.092]
- specificity delta +0.091, 95% CI [0.070, 0.125]
- specificity − target (= −non_target_mean_delta) +0.026, 95% CI
  [0.009, 0.035]; exact cluster sign-flip p = 0.25

Verdict: **suggestive, not confirmed.** The bootstrap CI excludes zero and the
direction is consistent across prompts (non-target emotion logits move down
under real steering), but with only 4 prompt clusters the exact permutation
test bottoms out at p = 0.125 and observed p = 0.25, so the design cannot
reject the noise hypothesis at any conventional level. Adjudication needs
replication (the 1.7B rerun, and/or multi-seed runs), not a bigger claim from
this run.

### Prototype 5.1 headline effect (run `20260708T061413Z__prototype51__...__98af1abdef`)

Row-level mean expected effect (primary scoring mode, real control, positive
strengths, KL ≤ 0.25): **+0.354** (the registered aggregate-level figure is
+0.372; the small difference is aggregation weighting).

- 95% bootstrap CI by pair (8 clusters): [0.120, 0.590]; by context (5
  clusters): [0.322, 0.382]
- exact pair-level sign-flip p = 0.055
- real − random paired contrast: +0.342, 95% CI [0.104, 0.572], exact p = 0.047
- real − wrong-emotion paired contrast: +0.486, 95% CI [0.024, 0.937]
- per-emotion CIs exclude zero for angry, happy, and sad; afraid is positive
  but its CI spans zero (fewest matched pairs).

The Prototype 5.1 positive result survives cluster-level uncertainty: the
effect is not driven by one activity pair or one context family.

## Length-normalized option scoring

`option_text_mean_logprob_margin` was added to `prototype51.py` as an **extra**
scoring mode: mean per-token logprob instead of the summed logprob, removing
option-length bias from the margin. `option_text_logprob_margin` remains the
pre-registered primary mode; the 0.6B config was not changed. The 1.7B
Prototype 5.1 config pre-registers the new mode as a diagnostic alongside the
original two.

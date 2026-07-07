# Prototype 5: emotion-steered activity preferences

Prototype 5 consumes the canonical Prototype 2.5 nested Prototype 1 vector bundle
and tests whether residual-stream emotion-vector steering changes compact
pairwise preferences over activities/actions.

The default bundle is:

```text
results/runs/colab-run-prototype-2.5/prototype1/20260628T205423Z__prototype1-prototype25__qwen-qwen3-0-6b-base__seed-42__8f5d56f306/
```

The runner inherits the measured base model from that bundle, defaulting to
`Qwen/Qwen3-0.6B-Base`, and uses Prototype 1's pre-registered selected layer
unless `intervention.layer` is set.

```bash
fe-prototype5 --config configs/prototype5.yaml
```

## Scope

Prototype 5 moves beyond emotion-word logits and generated-text lexical counts.
It asks whether compact emotion vectors have simple behavioral consequences:
when the model must choose between two activities, does steering shift the
next-token `A` versus `B` logit margin toward a preregistered activity category
for the target emotion?

The preregistered diagnostic hypotheses are:

- `happy`: social connection, celebration/play, helping/generosity, approach.
- `sad`: rest, withdrawal, reflection, low-energy actions.
- `angry`: confrontation, correction, boundary-setting.
- `afraid`: safety-seeking, checking, avoidance, caution.

These mappings are hypotheses for model-behavior diagnostics, not claims about
real subjective emotion.

Prototype 5 preserves this caveat in every summary:

> These are compact four-emotion vectors with weak valence/arousal geometry.
> Prototype 5 tests whether local causal steering transfers to simple preference
> behavior, not whether the model has mature human-like emotion structure.

## Preference Task

Each prompt is deterministic:

```text
The person is deciding what to do next.
A: {activity_a}
B: {activity_b}
The better next action is:
```

The primary score is the next-token logit margin `logit(A) - logit(B)`.
Prototype 5 records the baseline margin, steered margin, margin delta from
baseline, expected direction from the category mapping, and expected-direction
effect `expected_direction * margin_delta`.

The default sweep uses signed strengths:

```yaml
[-4, -2, -1, 0, 1, 2, 4]
```

## Controls and Diagnostics

Prototype 5 records:

- zero-steering fidelity against baseline logits;
- random-vector control with matched vector norm;
- wrong-emotion vector control;
- opposite-sign steering;
- KL divergence from baseline next-token distributions;
- pairwise preference scores for all activities, emotions, strengths, and controls;
- dose-response Spearman correlations for expected-direction effects;
- optional deterministic category Elo diagnostics as a compact summary.

## Outputs

Each run writes a timestamped directory under `artifacts/prototype5/`:

- `config.json`
- `metrics.json`
- `summary.json`
- `environment.json`
- `manifest.json`
- `diagnostics/preference_scores.json`
- `diagnostics/controls.json`
- `diagnostics/kl.json`
- `diagnostics/elo.json`

## Gates

Hard gates:

- required vector/model artifacts load successfully;
- zero steering reproduces baseline logits within
  `gates.zero_steering_max_abs_logit_error`;
- random-vector control is recorded;
- wrong-emotion control is recorded;
- KL diagnostics are recorded;
- pairwise preference scores are recorded.

Soft gates:

- positive steering shifts expected activity-category margins in the hypothesized
  direction;
- opposite-sign steering moves in the opposite direction;
- real-vector effect exceeds random-vector and wrong-emotion controls;
- dose-response Spearman is positive for expected preference effect;
- effects are stronger on matching emotion/category pairs than non-matching
  pairs.

# Prototype 4: causal emotion steering

Prototype 4 consumes the canonical Prototype 2.5 nested Prototype 1 vector bundle
and tests whether adding or subtracting saved emotion vectors at a selected
residual-stream layer causally changes model behavior in emotion-specific ways.

The default bundle is:

```text
results/runs/colab-run-prototype-2.5/prototype1/20260628T205423Z__prototype1-prototype25__qwen-qwen3-0-6b-base__seed-42__8f5d56f306/
```

The runner inherits the measured base model from that bundle, defaulting to
`Qwen/Qwen3-0.6B-Base`, and uses Prototype 1's pre-registered selected layer
unless `intervention.layer` is set.

```bash
fe-prototype4 --config configs/prototype4.yaml
```

## Scope

Prototype 3 found coherent enough vectors to analyze and steer, but did not find
a clean human-interpretable geometry: selected layer 19, four emotions, mean
absolute off-diagonal cosine around 0.33, weak valence/arousal alignment, and
mixed nearest neighbors.

Prototype 4 therefore preserves this caveat in every summary:

> These are compact four-emotion vectors with weak valence/arousal geometry.
> Prototype 4 tests local causal efficacy and specificity, not mature human-like
> emotion structure.

This prototype studies functional causal effects in model internals. It does not
make claims about subjective experience.

## Interventions

The experiment applies residual-stream steering hooks at the selected decoder
layer over signed strengths, defaulting to:

```yaml
[-4, -2, -1, 0, 1, 2, 4]
```

Two modes are recorded:

- Matching-token scoring: compact prompts compare next-token logits for
  emotion terms under real, random-vector, and wrong-emotion controls.
- Free generation: sampled continuations are generated under stronger signed
  steering strengths and scored with average token logprob, perplexity-like, and
  generated-text emotion-term diagnostics.

The default free-generation sweep is intentionally more aggressive than the
matching-token sweep: strengths `[-8, -4, 0, 4, 8]`, 96 generated tokens,
`temperature: 0.8`, `top_p: 0.95`, and three samples per condition over
emotion-open prompts such as `Avery read the note and felt`.

## Controls and Diagnostics

Prototype 4 records:

- zero-steering fidelity against baseline logits;
- random-vector control with matched vector norm;
- wrong-emotion vector control;
- opposite-sign steering;
- KL divergence from baseline next-token distributions;
- average generated-token logprob and perplexity-like fluency diagnostics;
- generated-text target emotion term counts and lexical specificity;
- specificity: target emotion terms versus non-target emotion terms;
- dose-response Spearman correlations for intended emotion effects.

## Outputs

Each run writes a timestamped directory under `artifacts/prototype4/`:

- `config.json`
- `metrics.json`
- `summary.json`
- `environment.json`
- `manifest.json`
- `diagnostics/matching_token_scores.json`
- `diagnostics/free_generation_samples.json`
- `diagnostics/kl_fluency.json`
- `diagnostics/controls.json`

## Gates

Hard gates:

- required vector artifacts load successfully;
- zero steering reproduces baseline logits within
  `gates.zero_steering_max_abs_logit_error`;
- random-vector control is recorded;
- KL diagnostics are recorded;
- fluency diagnostics are recorded.

Soft gates:

- intended emotion logit score improves over baseline for positive steering;
- opposite-sign steering moves in the opposite direction;
- intended effect exceeds random-vector and wrong-emotion controls;
- dose-response Spearman is positive for intended emotion effect;
- positive steering is more target-specific than non-target emotion movement.

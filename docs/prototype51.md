# Prototype 5.1: robust emotion-steered activity preferences

Prototype 5.1 is a robustness upgrade to Prototype 5. It consumes the same
canonical Prototype 2.5 nested Prototype 1 vector bundle and asks whether the
Prototype 5 null result survives better preference measurement.

The default bundle is:

```text
results/runs/colab-run-prototype-2.5/prototype1/20260628T205423Z__prototype1-prototype25__qwen-qwen3-0-6b-base__seed-42__8f5d56f306/
```

Run it with:

```bash
fe-prototype51 --config configs/prototype51.yaml
```

## Scope

Prototype 5.1 is not a new extraction. It tests whether the original Prototype 5
activity-preference null was caused by brittle A/B token scoring, option-order
bias, missing context, or layer/strength choice.

It preserves this caveat in every summary:

> These are compact four-emotion vectors with weak valence/arousal geometry.
> Prototype 5.1 tests whether the Prototype 5 null result survives more robust
> preference measurement, not whether the model has mature human-like emotion
> structure.

## Robustness Upgrades

For every activity pair, Prototype 5.1 evaluates both original and swapped
option order. Expected-direction effects are aggregated over both orders so
literal `A`/`B` and position priors are visible instead of silently absorbed.

The runner records two scoring modes by default:

- `choice_token_margin`: the Prototype 5-style next-token `logit(A) - logit(B)`.
- `option_text_logprob_margin`: conditional log probability of each full option
  text after `{context}\nThe better next action is`.

The default contexts are deterministic diagnostic families: neutral,
positive/social, loss/low-energy, conflict, and safety/uncertainty. They are
prompt families for robustness checks, not ground-truth emotional labels.

The default layer sweep is:

```yaml
[16, 17, 18, 19, 20, 21, 22]
```

The default signed strength sweep is:

```yaml
[-12, -8, -4, -2, 0, 2, 4, 8, 12]
```

## Metrics

Prototype 5.1 reports aggregate and KL-filtered effects by emotion, context
family, scoring mode, layer, strength, and control. The primary summary uses
`option_text_logprob_margin` with `gates.kl_max_for_effect_summary` as a
guardrail, defaulting to `0.25`.

Controls include zero steering, opposite-sign steering, matched-norm random
vectors, wrong-emotion vectors, KL diagnostics, and zero-steering fidelity for
both next-token logits and logprob/margin scores.

The runner reports the best diagnostic layer by mean positive real expected
effect under the primary scoring mode. This is a diagnostic report, not a hard
gate.

## Outputs

Each run writes a timestamped directory under `artifacts/prototype51/`:

- `config.json`
- `metrics.json`
- `summary.json`
- `environment.json`
- `manifest.json`
- `diagnostics/preference_scores.json`
- `diagnostics/order_swap_scores.json`
- `diagnostics/option_text_scores.json`
- `diagnostics/layer_sweep.json`
- `diagnostics/controls.json`
- `diagnostics/kl.json`
- `diagnostics/elo.json`

## Gates

Hard gates:

- required vector/model artifacts load successfully;
- at least one configured layer has all required emotion vectors;
- zero steering reproduces baseline logits and logprobs within tight tolerance;
- order-swap diagnostics are recorded;
- option-text scoring diagnostics are recorded;
- random-vector control is recorded;
- wrong-emotion control is recorded;
- KL diagnostics are recorded;
- pairwise preference scores are recorded.

Soft gates:

- positive steering shifts expected activity-category margins under option-text
  scoring;
- order-swapped aggregation has the same sign as unswapped scoring or exposes
  reduced order bias;
- opposite-sign steering moves in the opposite direction;
- real-vector effect exceeds random-vector and wrong-emotion controls;
- dose-response Spearman is positive;
- matching emotion/category pairs are stronger than non-matching pairs;
- at least one emotion, especially `happy`, shows a positive robust effect
  under the KL guardrail.

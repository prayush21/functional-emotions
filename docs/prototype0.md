# Prototype 0 experiment card

## Question

Can we reliably observe and causally intervene on the residual stream of an
open-weight causal language model before attempting to reproduce the paper's
emotion-vector results?

## Hypotheses

### H0 — instrumentation integrity

A hook at a selected decoder block captures a tensor with shape
`[batch, sequence, hidden]`. Applying the same hook with strength zero produces
logits identical to an unhooked forward pass.

### H1 — causal controllability

Adding a non-zero direction at that block changes downstream next-token logits,
and the magnitude of the change generally increases with the absolute steering
strength.

### H2 — semantic smoke test

A normalized difference between activations from positive and negative prompts
should move a `happy`-versus-`sad` next-token logit margin in the corresponding
direction. This is diagnostic only: four prompt pairs are insufficient to call
the direction an emotion representation.

## Model and compute

The canonical local run uses `Qwen/Qwen3-0.6B-Base`, float32, on Apple MPS. It
fits the project's 8 GB M2 machine. The cloud runner uses
`Qwen/Qwen3-1.7B-Base`, float16, on a Modal T4. Free Colab is also sufficient for
either model.

Required software is Python 3.11+, PyTorch, Transformers, Safetensors, NumPy,
PyYAML, pytest, and Ruff. No API key is required for the public Qwen models,
though a Hugging Face token improves download rate limits.

## Procedure

1. Load the base model and resolve its middle decoder block.
2. Run four positive and four negative calibration prompts.
3. Capture each prompt's last-token block output.
4. Compute and normalize the difference between positive and negative means.
5. Run the neutral prompt `After thinking about what happened, I feel` without
   intervention.
6. Add the direction at every prompt-token position at signed strengths from
   -0.1 to +0.1, expressed as fractions of the baseline residual norm.
7. Record exact no-op error, logit L2 distance, KL divergence, effect-strength
   monotonicity, and the ` happy` minus ` sad` logit margin.
8. Save the direction, resolved configuration, environment, and metrics.

## Hard gates

- Zero-steering maximum absolute logit error ≤ `1e-5`.
- At least one non-zero intervention has logit L2 distance ≥ `1e-4`.
- Spearman correlation between absolute strength and effect magnitude ≥ `0.8`.

The semantic margin is deliberately not a gate. Prototype 0 validates the
apparatus; Prototype 1 validates emotion representations.

## Canonical local result

Run: `20260619T223416Z`

- All three hard gates passed.
- Zero-steering logit error: `0.0`.
- Hidden size: `1024`; selected layer: `14`.
- Strength/effect Spearman correlation: `0.943`.
- Baseline happy-minus-sad margin: `-1.933`.
- At `-0.1`, the margin moved to `-3.762`.
- At `+0.1`, the margin moved to `+0.262`.
- KL divergence remained modest across the paper-scale sweep: `0.065` at
  `-0.1` and `0.123` at `+0.1`.

This supports H0 and H1 and gives a positive smoke-test result for H2. It does
not establish a general emotion vector, semantic selectivity, or behavioral
relevance.

## Expected failures and interpretation

- If zero steering changes logits, the hook implementation is invalid.
- If non-zero steering has no effect, the wrong module may be hooked or the
  direction may have a dtype/device mismatch.
- If effect size is non-monotonic only at large strengths, the model may be in a
  nonlinear or degraded regime; reduce the sweep.
- If logits change but the semantic margin does not, the apparatus still works;
  the small calibration set simply did not identify a semantic direction.
- If the semantic margin moves while KL explodes, the result is not useful
  steering—it is model disruption.


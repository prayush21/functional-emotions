# Functional Emotions

An open-weight replication of the methods in Anthropic's *Emotion Concepts and
their Function in a Large Language Model*. This project studies functional,
causal representations of emotion concepts. It does **not** treat those
representations as evidence of subjective experience.

## Prototype 0: intervention plumbing

Prototype 0 tests one deliberately narrow claim:

> Residual-stream activations can be captured at a chosen transformer layer and
> changed by a controlled vector intervention, with a zero-strength intervention
> exactly reproducing the unmodified model and non-zero strengths producing a
> measurable dose-dependent change in next-token predictions.

It uses paired positive/negative calibration prompts to construct a rough
difference-in-means direction. That direction is only an instrumentation fixture;
it is **not yet** a validated emotion vector. The experiment steers a neutral
prompt over signed strengths and records:

- maximum baseline-versus-zero-steering logit error;
- residual activation shape and norm;
- KL divergence from baseline;
- L2 logit change;
- a diagnostic `happy` versus `sad` next-token logit margin;
- monotonicity of effect magnitude with absolute steering strength.

### Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
fe-prototype0 --config configs/prototype0.yaml
pytest
```

The default `Qwen/Qwen3-0.6B-Base` run is intended for an 8 GB Apple Silicon
machine. Model download is approximately a low-single-digit number of GB. CPU is
supported but slower; Apple Silicon uses MPS automatically when available.

### Outputs

Each run creates a timestamped directory under `artifacts/prototype0/` containing:

- `config.json` — fully resolved run configuration;
- `metrics.json` — pass/fail gates and sweep measurements;
- `direction.safetensors` — the derived intervention direction;
- `environment.json` — package, platform, device, and model revision metadata.
- `manifest.json` — stable run ID plus exact model and code revisions;
- `summary.json` — compact metrics for experiment comparison.

Full artifacts are intentionally ignored by Git. Register an accepted local or
downloaded cloud run in the lightweight, version-controlled result registry:

```bash
fe-register-run /path/to/completed-run
```

See [results/README.md](results/README.md) for the registry format and
[docs/experiment_tracking.md](docs/experiment_tracking.md) for the complete
Git-plus-durable-storage workflow.

### Colab and Modal

For the 0.6B model, local execution or free Colab is the cheapest path. For the
1.7B model, use a free/low-cost Colab T4 when available. A dedicated configuration
keeps the Colab run reproducible:

```bash
fe-prototype0 --config configs/prototype0_qwen3_1.7b_colab.yaml
```

The full Colab walkthrough, including cloning and downloading artifacts, is in
[docs/colab.md](docs/colab.md).

Modal is useful for repeatable short GPU jobs and should be run only when a GPU
is needed:

```bash
modal run cloud/modal_prototype0.py
```

Prototype 0 is complete when all hard gates in `metrics.json` pass. Semantic
movement of the happy/sad margin is reported as a diagnostic, not a hard gate.

## Prototype 1: emotion-vector extraction

Prototype 1 implements the paper's core extraction pipeline at a deliberately
compact scale: implicit-emotion story generation, held-out topic splits,
per-layer difference-in-means directions, neutral-corpus PCA removal, and
held-out classification validation.

```bash
fe-prototype1 --config configs/prototype1.yaml --stage generate
fe-prototype1 --config configs/prototype1.yaml --stage extract
```

The generator and measured base model are configured separately. Runs record
dataset hashes, the exact topic split, raw-versus-cleaned validation at every
layer, and safetensor bundles for both emotion vectors and neutral components.
Prototype 1 now runs end to end. The first completed compact run found
above-chance held-out signal, but failed the mean-margin hard gate because the
uncalibrated multiclass scores over-predicted `angry` and never selected
`afraid` at the pre-registered layer. Prototype 2 will test semantic robustness
and controls rather than tuning Prototype 1 to pass.
See [docs/prototype1.md](docs/prototype1.md) for the experiment card, gates,
scaling guidance, and interpretation limits.

## Prototype 2: semantic validation and controls

Prototype 2 explains the Prototype 1 result rather than tuning it to pass. It
loads a completed Prototype 1 artifact bundle, reruns held-out scoring, compares
real versus shuffled vector labels, raw versus PCA-cleaned directions, diagnostic
best layers, emotion confusion patterns, compact lexical scenarios,
topic-stratified held-out metrics, logit-lens emotion-word effects, implicit
intensity sweeps, and simple train-score calibration.

```bash
fe-prototype2 --config configs/prototype2.yaml
```

See [docs/prototype2.md](docs/prototype2.md) for Colab cells, expected outputs,
and interpretation guidance.

## Prototype 2.5: revised extraction before geometry

Prototype 2.5 is a bridge step prompted by Prototype 2's failure modes. It
orchestrates a larger, better-balanced Prototype 1 extraction and immediately
runs the Prototype 2 controls on the new bundle.

```bash
fe-prototype25 --config configs/prototype25.yaml --stage prepare
fe-prototype25 --config configs/prototype25.yaml --stage all
```

Use this before Prototype 3 if semantic controls fail on the current vectors.
See [docs/prototype25.md](docs/prototype25.md).

## Prototype 3: emotion-space geometry

Prototype 3 consumes the canonical Prototype 2.5 vector bundle and measures
diagnostic geometry: cosine structure, nearest neighbors, clustering, PCA,
valence/arousal alignment, and representational similarity across layers.

```bash
fe-prototype3 --config configs/prototype3.yaml
```

This is a structural checkpoint before causal steering. It does not rerun
extraction or semantic validation, and it does not make claims about subjective
experience. See [docs/prototype3.md](docs/prototype3.md).

## Prototype 4: causal emotion steering

Prototype 4 consumes the canonical Prototype 2.5 nested Prototype 1 vector
bundle and applies signed residual-stream interventions at the selected layer.
It records matching-token and free-generation effects with dose-response,
specificity, random-vector, wrong-emotion, KL, and fluency controls.

```bash
fe-prototype4 --config configs/prototype4.yaml
```

Because Prototype 3 found weak valence/arousal geometry, Prototype 4 is framed
narrowly as local causal efficacy over compact validated vectors, not evidence
of a mature emotion manifold. See [docs/prototype4.md](docs/prototype4.md).

## Prototype 5: activity preferences

Prototype 5 tests whether the same compact emotion-vector steering transfers to
simple pairwise activity preferences. It scores deterministic `A` versus `B`
next-token logit margins for activities such as calling a friend, resting,
checking for safety, correcting someone, helping, or joining a celebration.

```bash
fe-prototype5 --config configs/prototype5.yaml
```

The default run records zero-steering fidelity, expected-direction preference
effects, dose-response, opposite-sign reversal, random-vector and wrong-emotion
controls, KL diagnostics, and lightweight category Elo summaries. It preserves
the Prototype 3/4 caveat: this tests local causal preference consequences, not
mature human-like emotion structure. See [docs/prototype5.md](docs/prototype5.md).

## Prototype 5.1: robust activity-preference assay

Prototype 5.1 keeps the same canonical Prototype 2.5 vector bundle and asks
whether Prototype 5's null behavioral-transfer result survives a stronger
assay. It adds A/B order swaps, full option-text logprob scoring, contextualized
prompt families, per-emotion/context/scoring/layer breakdowns, a compact layer
sweep around layer 19, stronger signed strengths, and KL guardrails.

```bash
fe-prototype51 --config configs/prototype51.yaml
```

This is not a new extraction step. It tests whether the Prototype 5 activity
preference null was caused by brittle A/B token scoring, option-order bias,
missing context, or layer/strength choice. See [docs/prototype51.md](docs/prototype51.md).

## Roadmap

See [docs/roadmap.md](docs/roadmap.md).

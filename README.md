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
See [docs/prototype1.md](docs/prototype1.md) for the experiment card, gates,
scaling guidance, and interpretation limits.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md).

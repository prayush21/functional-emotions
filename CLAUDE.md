# CLAUDE.md

Open-weight replication of Anthropic's *Emotion Concepts and their Function in
a Large Language Model*. The project extracts functional, causal emotion-vector
representations from small Qwen3 models and tests their effects via activation
steering. Work is organized as a gated ladder of "prototypes" (0 through 5.1,
with 6 and 7 planned); each prototype either passes its pre-registered hard
gates or reports a failure that motivates the next prototype. See
`docs/roadmap.md` for the full ladder and current status of each stage.

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Checks
pytest
ruff check .

# Local experiment (N = 0, 1, 2, 25, 3, 4, 5, 51)
fe-prototype<N> --config configs/prototype<N>.yaml
# prototype1 and prototype25 additionally require --stage (e.g. generate,
# extract, prepare, all)
fe-prototype1 --config configs/prototype1.yaml --stage extract

# Cloud loop (Modal), example for prototype 5.1 on a T4
modal run cloud/modal_run.py --prototype 51 --config configs/prototype51.yaml --gpu T4
python cloud/fetch_run.py --prototype 51 --register
```

## Architecture

- `src/functional_emotions/instrumentation.py` — activation capture and
  steering hooks (the shared intervention plumbing every prototype builds on).
- `src/functional_emotions/tracking.py` — run manifests, stable run IDs, and
  provenance metadata (model revision, code commit, config hash).
- `src/functional_emotions/registry.py` — the `fe-register-run` result
  registry implementation.
- `src/functional_emotions/prototype<N>.py` — self-contained experiment
  runners, one per prototype, each exposed as a `fe-prototype<N>` console
  script.
- `artifacts/` — full run outputs (configs, metrics, vectors, environment
  metadata). Gitignored; large and reproducible, not committed.
- `results/` — the version-controlled experiment registry: `index.jsonl` (one
  compact JSON record per accepted run) plus `results/runs/<run-id>/` (small
  reviewable metadata/metrics copies). This is the durable research ledger.
- `configs/` — YAML configuration per prototype, with pre-registered hard
  gates baked into each config. Configs are the experiment's pre-registration;
  see `docs/experiment_tracking.md` for the full tracking workflow.

## Standing research rules

- Never tune a prototype's config or code to make its hard gates pass. A
  failed gate is a result, not a bug to patch away — investigate it in the
  *next* prototype instead of editing the current one to force a pass.
- Every accepted run must be registered with `fe-register-run` (or the cloud
  `--register` step) before conclusions are drawn from it. Unregistered runs
  are not part of the research record.
- When summarizing results, preserve the interpretation caveat: these are
  functional/causal findings about model internals and behavior, not evidence
  of subjective experience.
- Tests must keep passing without downloading model weights. Do not add tests
  that require network access or a real Hugging Face model download.

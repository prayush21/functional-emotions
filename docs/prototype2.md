# Prototype 2 experiment card

## Question

Do the Prototype 1 directions reflect semantic emotion signal, or are they mostly
artifacts of topic, lexical leakage, generation style, vector cleaning, or score
scale?

Prototype 2 is not a rescue attempt for Prototype 1. It keeps the Prototype 1
pre-registered layer and failed margin result intact, then adds controls that
explain what the signal appears to be.

## Method

Prototype 2 consumes a completed Prototype 1 artifact bundle:

- `emotion_vectors.safetensors`
- `neutral_pca.safetensors`
- `dataset/stories.jsonl`
- `metrics.json`
- `config.json`

It recomputes held-out activations for the saved stories, reconstructs raw
difference-in-means vectors from the saved train split, and compares those raw
vectors with the saved PCA-cleaned vectors. It also evaluates shuffled vector
labels, layer sweeps, confusion patterns, compact lexical scenarios, held-out
topic-stratified generalization, logit-lens emotion-word scores, implicit
intensity levels, and a train-score z-score calibration diagnostic.

The layer sweep is diagnostic only. It does not replace Prototype 1's
pre-registered selected layer.

## Run

```bash
fe-prototype2 --config configs/prototype2.yaml
```

In Colab:

```bash
git pull
pip install -e ".[dev]"
fe-prototype2 --config configs/prototype2.yaml
python - <<'PY'
import json
from pathlib import Path
latest = sorted(Path("artifacts/prototype2").glob("*"))[-1]
print(latest)
print(json.dumps(json.loads((latest / "summary.json").read_text()), indent=2))
PY
```

If your Prototype 1 bundle lives elsewhere, edit `prototype1.run_dir` in
`configs/prototype2.yaml` before running.

## Soft Gates

- Real labels should beat shuffled labels on held-out macro AUC and/or accuracy.
- The compact lexical robustness set should beat chance.
- The mean intensity Spearman correlation should be positive.
- Topic-stratified held-out metrics and logit-lens effects should be inspected
  as diagnostics, not replacement gates.
- Calibration should report whether the margin failure looks like score-scale
  mismatch. It is diagnostic and does not rewrite Prototype 1.

## Outputs

Each timestamped directory under `artifacts/prototype2/` contains:

- `config.json`, `metrics.json`, `summary.json`, `environment.json`, and
  `manifest.json`;
- `dataset/stories.jsonl`, `dataset/lexical_scenarios.jsonl`, and
  `dataset/intensity_scenarios.jsonl`;
- `diagnostics/lexical_scores.json`, `diagnostics/intensity_scores.json`,
  `diagnostics/logit_lens.json`, and
  `diagnostics/cross_topic_generalization.json`.

The `summary.json` answers the guiding questions directly: shuffled-label signal,
layer stability, topic generalization, logit-lens signal, PCA effect, `afraid`
argmax wins, `angry` dominance, intensity monotonicity, lexical robustness,
calibration effect, and whether Prototype 3 geometry is warranted.

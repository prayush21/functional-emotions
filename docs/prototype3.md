# Prototype 3 experiment card

## Question

Do the canonical Prototype 2.5 emotion vectors have interpretable geometry across
layers?

Prototype 3 is diagnostic geometry. It measures structure in the saved vectors
before moving to causal steering. It does not introduce a new extraction method,
rerun semantic validation, or claim a mature paper-scale emotion map.

## Input

The default config consumes the canonical Prototype 2.5 nested Prototype 1
bundle:

```text
results/runs/colab-run-prototype-2.5/prototype1/20260628T205423Z__prototype1-prototype25__qwen-qwen3-0-6b-base__seed-42__8f5d56f306/
```

The key input is:

```text
emotion_vectors.safetensors
```

Prototype 3 records the matching nested Prototype 2 validation run as provenance
but does not rerun those controls.

## Method

For every saved layer, `fe-prototype3` computes:

- emotion-by-emotion cosine similarity;
- nearest neighbors for each emotion;
- average-link agglomerative clustering over cosine similarity;
- two-dimensional PCA coordinates and explained variance;
- alignment between vector cosine structure and compact valence/arousal anchors;
- representational similarity between each layer's cosine structure and the
  pre-registered selected layer.

UMAP is recorded as unavailable because it is not currently a project dependency.
PCA provides the committed projection diagnostic; UMAP can be added later as an
optional visualization dependency if needed.

## Run

```bash
fe-prototype3 --config configs/prototype3.yaml
```

Each timestamped directory under `artifacts/prototype3/` contains:

- `config.json`, `metrics.json`, `summary.json`, `environment.json`, and
  `manifest.json`;
- `diagnostics/selected_layer_geometry.json`;
- `diagnostics/representational_similarity.json`;
- `diagnostics/pca_by_layer.json`.

## Interpretation

Prototype 3 should be read as a structural checkpoint:

```text
Do the improved compact vectors have coherent geometry worth steering?
```

It should not be read as:

```text
The model has a complete human-like emotion space.
```

The current setup is still a four-emotion synthetic bridge run. Strong geometry
would support moving to Prototype 4 causal steering over these same vectors.
Weak or uninterpretable geometry would suggest revisiting extraction scale,
emotion coverage, or valence/arousal anchors before causal claims.

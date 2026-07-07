# Prototype 1 experiment card

## Question

Can a small open-weight base model yield linear emotion-concept directions that
generalize to stories about topics never used to construct those directions?

Prototype 1 is an extraction and held-out validation experiment. It does not yet
claim lexical robustness, causal influence, or human-like emotion geometry; those
belong to later prototypes.

## Method

The implementation follows the paper's extraction recipe at reduced scale:

1. An instruction-tuned generator writes implicit-emotion stories for every
   `(topic, emotion)` pair. The target word and configured direct synonyms are
   forbidden and automatically audited.
2. Topics are deterministically split into train and test sets. Every story for a
   topic stays in one split, preventing near-duplicate premises from crossing the
   boundary.
3. The base model processes each story. Residual-stream activations are averaged
   over tokens beginning at token 50 for every decoder layer. A shorter story
   contributes its final-token activation rather than an empty average.
4. For emotion `e`, the raw direction is the mean train activation for `e` minus
   the mean activation across all train emotions.
5. On separate emotionally neutral paragraphs, PCA components sufficient to
   explain 50% of activation variance are computed independently at each layer.
   Those components are projected out of every raw direction, and the result is
   normalized.
6. Cleaned vectors are scored on held-out-topic stories using dot products.
   Accuracy, one-vs-rest macro AUC, correct-class margin, and per-emotion metrics
   are reported for every layer.

The hard-gate layer is pre-registered as two-thirds of the way through the model,
matching the paper's main analysis depth. The code does not select the best layer
using held-out results.

## Run

Generate the compact dataset, then extract and validate vectors:

```bash
fe-prototype1 --config configs/prototype1.yaml --stage generate
fe-prototype1 --config configs/prototype1.yaml --stage extract
```

Or run both stages in one process:

```bash
fe-prototype1 --config configs/prototype1.yaml --stage all
```

Generation and extraction intentionally use separate model settings. The default
generator is instruction-tuned; the measured model is `Qwen/Qwen3-0.6B-Base`.
The committed config is a 4-emotion, 12-topic feasibility run. A paper-scale run
should expand to the paper's 171 emotions, 100 topics, and 12 stories per pair
only after this version passes and artifact storage is ready.

## Hard gates

- Train and held-out topic sets are disjoint.
- No story contains its emotion label or configured direct synonyms.
- Held-out accuracy at the pre-registered layer is at least `0.40` versus `0.25`
  chance for four emotions.
- Held-out macro AUC is at least `0.60`.
- Mean correct-class margin is non-negative.

Thresholds are feasibility criteria, not evidence that the replication is
complete. Prototype 2 adds lexical controls, shuffled-label controls, intensity
sweeps, implicit scenarios, and logit-lens validation.

## Current result

A completed compact run on June 27, 2026 did not pass all hard gates:

- held-out accuracy was `0.472`, above the `0.40` feasibility threshold;
- held-out macro AUC was `0.867`, above the `0.60` feasibility threshold;
- mean correct-class margin was `-3.261`, below the non-negative margin gate.

The failure mode was not topic leakage, label leakage, or token-pooling fallback.
Train and held-out topic sets were disjoint, the lexical audit passed, and no
train or held-out stories used final-token fallback pooling. The main issue was
multiclass calibration at the pre-registered layer: `angry` was over-predicted
and `afraid` was never the argmax, despite one-vs-rest AUC indicating separable
signal. Prototype 1 should therefore be treated as a completed failed
feasibility run: it found emotion-related signal, but the cleaned
difference-in-means vectors were not a reliable four-way classifier.

## Outputs

Each timestamped directory under `artifacts/prototype1/` contains:

- `emotion_vectors.safetensors` — cleaned vector for each emotion and layer;
- `neutral_pca.safetensors` — per-layer neutral principal components;
- `metrics.json` — gates, dataset audit, split, PCA metadata, and validation;
- `dataset/stories.jsonl` and `dataset/neutral.jsonl` — the exact generated text
  used for extraction and validation;
- `config.json`, `environment.json`, `manifest.json`, and `summary.json` — exact
  provenance and compact registry data.

The environment record includes SHA-256 hashes of both JSONL datasets. The metrics
also include pooling diagnostics, confusion matrices for every layer, and
selected-layer per-example score dumps so failures can be inspected after a Colab
runtime shuts down. Register an accepted run with `fe-register-run` using the same
workflow as Prototype 0.

## Expected failures

- Leakage errors mean the generated text named the target concept; regenerate or
  extend `forbidden_terms`.
- Chance held-out accuracy with strong train separation indicates topic/template
  confounding or a direction that does not generalize.
- High macro AUC with negative correct-class margin means one-vs-rest ranking is
  promising, but the multiclass winner is still wrong often enough to fail the
  hard gate. Inspect `selected_layer_examples` and the confusion matrix.
- PCA removal making performance worse is reportable. Raw and cleaned metrics are
  both retained; the implementation never hides the comparison.
- Very short generations undermine token-50 pooling. Increase generation length
  before interpreting a run that frequently uses the final-token fallback reported
  under `pooling`.

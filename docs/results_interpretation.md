# Result interpretation notes

## Current canonical handoff

The current canonical vector source for Prototype 3 is the Prototype 1 extraction
nested inside:

```text
results/runs/colab-run-prototype-2.5/
```

Specifically:

```text
results/runs/colab-run-prototype-2.5/prototype1/20260628T205423Z__prototype1-prototype25__qwen-qwen3-0-6b-base__seed-42__8f5d56f306/
```

This bundle contains the vectors and PCA components Prototype 3 should consume:

```text
emotion_vectors.safetensors
neutral_pca.safetensors
dataset/stories.jsonl
dataset/neutral.jsonl
metrics.json
summary.json
config.json
```

## Why not the original Prototype 1 vectors?

The original Prototype 1 run is preserved as a failed-but-informative baseline:

```text
results/runs/20260623T232651Z__prototype1__qwen-qwen3-0-6b-base__seed-42__f0d38f143e/
```

It found real held-out signal, but failed the mean-margin hard gate because
`angry` was over-predicted and `afraid` did not win argmax at the selected layer.
That failure motivated Prototype 2 semantic controls and Prototype 2.5's revised
extraction.

## What Prototype 2.5 established

Prototype 2.5 revised the compact extraction setup before geometry:

- 4 emotions;
- 16 topics;
- 4 stories per topic/emotion;
- 16 neutral topics;
- more explicit topic balance between `angry` and `afraid` contexts.

The nested Prototype 1 extraction passed all hard gates:

```text
held_out_accuracy: 0.6125
held_out_macro_auc: 0.90375
mean_correct_margin: 3.80739
```

The nested Prototype 2 validation passed all soft gates:

```text
shuffled_macro_auc_effect: 0.51521
lexical_accuracy: 0.50
intensity_mean_spearman: 0.125
minimum_topic_accuracy: 0.50
```

This changes the research status from:

```text
real signal, but too fragile for geometry
```

to:

```text
usable compact emotion vectors, ready for diagnostic geometry
```

## Caveats

Prototype 2.5 does not establish a mature or general emotion space. It is still a
compact four-emotion synthetic setup. The intensity sweep is only weakly
positive, and lexical robustness is based on a compact hand-authored set.

Prototype 3 should therefore be framed as diagnostic geometry: measuring whether
the improved vectors have interpretable structure, not claiming a complete map
of emotion representations.

## Registry roles

`results/index.jsonl` marks the important runs with roles:

- `failed_baseline_for_semantic_validation` — original Prototype 1;
- `current_best_orchestrator_and_prototype3_handoff` — Prototype 2.5 wrapper;
- `canonical_vector_source_for_prototype3` — nested Prototype 1 vector bundle;
- `semantic_validation_for_canonical_vectors` — nested Prototype 2 controls.

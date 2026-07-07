# Prototype 2.5 Colab Run

This is the current accepted handoff bundle for Prototype 3.

## Contents

- `config.json`, `metrics.json`, `summary.json`, `environment.json`, and
  `manifest.json` describe the Prototype 2.5 orchestrator run.
- `dataset/` contains the generated Prototype 2.5 story and neutral datasets.
- `prototype1/20260628T205423Z__prototype1-prototype25__qwen-qwen3-0-6b-base__seed-42__8f5d56f306/`
  contains the canonical vector source for Prototype 3.
- `prototype2/20260628T205450Z__prototype2-prototype25-validation__qwen-qwen3-0-6b-base__seed-42__39a92c5ae9/`
  contains the semantic validation controls for those vectors.

## Prototype 3 Input

Use this nested Prototype 1 bundle:

```text
results/runs/colab-run-prototype-2.5/prototype1/20260628T205423Z__prototype1-prototype25__qwen-qwen3-0-6b-base__seed-42__8f5d56f306/
```

The key files are:

```text
emotion_vectors.safetensors
neutral_pca.safetensors
dataset/stories.jsonl
dataset/neutral.jsonl
metrics.json
summary.json
config.json
```

## Interpretation

This run revised the compact extraction setup after Prototype 2 found fragile
semantic robustness. The nested Prototype 1 extraction passed hard gates, and the
nested Prototype 2 validation passed semantic soft gates. Prototype 3 should
therefore proceed as diagnostic geometry over these vectors.

# Experiment registry

This directory stores lightweight, reviewable records of completed experiments.
Full run bundles remain under `artifacts/` or in external storage because they
may contain large activation tensors and vectors.

Register a downloaded Colab bundle from the repository root:

```bash
fe-register-run /path/to/prototype0-results
```

The command validates the bundle, creates a stable run ID, copies only JSON
metadata and metrics into `results/runs/<run-id>/`, and appends a compact record
to `results/index.jsonl`. Re-registering the same bundle is idempotent.

Commit `results/` to Git when a run is accepted as part of the research record.
Keep the original zipped bundle in durable object storage or Google Drive so
large files such as `direction.safetensors` are not lost.

New runs record:

- the exact Hugging Face model commit SHA;
- the requested model revision;
- the exact repository commit and whether the tree was dirty;
- a canonical configuration hash;
- package, platform, device, and seed metadata;
- compact outcome metrics and gates.


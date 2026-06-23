# Experiment tracking workflow

Use two storage tiers for every accepted run.

## Full artifact bundle

Keep the complete timestamped run directory in durable storage such as Google
Drive, an object-storage bucket, or a backed-up research drive. This bundle may
contain large files such as `direction.safetensors`, activation caches, datasets,
or generated transcripts, so it is intentionally excluded from Git.

Recommended layout:

```text
functional-emotions-artifacts/
  prototype0/
    qwen3-0.6b-base/
      <run-id>.zip
    qwen3-1.7b-base/
      <run-id>.zip
  prototype1/
    <model>/
      <run-id>.zip
```

Never rename files inside a bundle. The manifest and run ID are the identity of
the experiment.

## Lightweight Git registry

After downloading or copying a completed bundle into a local directory, run:

```bash
fe-register-run /path/to/completed-run
```

This copies only small, reviewable records into `results/runs/<run-id>/` and
adds one line to `results/index.jsonl`. Commit those files alongside the code.
The registry is the searchable experiment ledger; durable storage holds the
large underlying evidence.

## Colab with Google Drive

After the experiment finishes and `latest` points to its artifact directory:

```python
from google.colab import drive
import shutil

drive.mount("/content/drive")
destination = "/content/drive/MyDrive/functional-emotions-artifacts/prototype0/qwen3-1.7b-base"
archive = shutil.make_archive(f"{destination}/{latest.name}", "zip", latest)
print(archive)
```

Then register the unzipped bundle locally and commit the resulting `results/`
changes. If a full artifact archive is lost, the Git registry preserves the
metrics and provenance, but not learned vectors or large raw data.

## Comparing runs

`results/index.jsonl` contains one compact JSON object per run, including model,
layer, hidden size, gates, endpoint margins, exact model revision, and code
commit. It can be loaded with pandas:

```python
import pandas as pd

runs = pd.read_json("results/index.jsonl", lines=True)
runs[["experiment", "model_name", "all_hard_gates_pass", "baseline_margin"]]
```


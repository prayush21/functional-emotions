# Prototype 2.5 experiment card

## Question

Can a revised compact extraction run fix the specific weaknesses Prototype 2
found before we move to emotion-space geometry?

Prototype 2.5 is a bridge between semantic validation and geometry. It does not
introduce a new representation method. Instead, it runs a better-balanced
Prototype 1 extraction and immediately validates the resulting vectors with the
Prototype 2 controls.

## Motivation

Prototype 2 found signal beyond shuffled labels, but weak lexical robustness,
negative intensity monotonicity, `angry` over-prediction, and poor selected-layer
`afraid` wins. That means Prototype 3 geometry would be premature on the current
vectors. Prototype 2.5 tests whether a modest data/extraction revision improves
those failure modes.

## Method

`fe-prototype25` orchestrates existing code:

1. Resolve a larger, balanced Prototype 1 config.
2. Generate implicit stories with more topics and more stories per
   topic/emotion.
3. Extract raw and PCA-cleaned emotion vectors.
4. Run Prototype 2 semantic controls on the new Prototype 1 bundle.
5. Write an orchestrator artifact with the exact resolved configs and links to
   the nested Prototype 1 and Prototype 2 outputs.

The default config keeps the run Colab-sized:

- 4 emotions;
- 16 topics;
- 4 stories per topic/emotion;
- 16 neutral topics;
- explicit topic contrasts intended to separate `angry` from `afraid`.

## Run

Dry-run the resolved configs without loading models:

```bash
fe-prototype25 --config configs/prototype25.yaml --stage prepare
```

Run the full Colab pipeline:

```bash
fe-prototype25 --config configs/prototype25.yaml --stage all
```

Stages are available separately:

```bash
RUN_DIR=artifacts/prototype25/my-run
fe-prototype25 --config configs/prototype25.yaml --stage prepare --run-dir "$RUN_DIR"
fe-prototype25 --config configs/prototype25.yaml --stage generate --run-dir "$RUN_DIR"
fe-prototype25 --config configs/prototype25.yaml --stage extract --run-dir "$RUN_DIR"
fe-prototype25 --config configs/prototype25.yaml --stage validate --run-dir "$RUN_DIR"
```

For `--stage validate`, set `prototype2.prototype1_run_dir` in
`configs/prototype25.yaml` to an existing Prototype 1 run directory unless the
same staged run already completed `--stage extract`.

## Decision Rule

Proceed to Prototype 3 only if the nested Prototype 2 validation shows:

- real labels beat shuffled labels;
- lexical robustness beats chance;
- mean intensity Spearman is positive;
- `angry` dominance and selected-layer `afraid` failure are reduced.

Otherwise revise generation/extraction again before geometry.

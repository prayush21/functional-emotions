# Prototype 6: representation dynamics (experiment card — DESIGN, NOT YET APPROVED)

Status: design draft awaiting approval (2026-07-11). No Prototype 6 code exists
and none will be written until this card is approved. The config sketched here
is the intended pre-registration; once approved it is frozen like every other
prototype config.

Prototype 6 asks *what* the emotion vectors track, not just whether they exist
and steer: whose emotion is represented, whether negation flips or erases it,
whether it persists across intervening context, and whether instruction tuning
moves it.

It preserves this caveat in every summary:

> These are compact four-emotion vectors with weak valence/arousal geometry
> and (at 1.7B) weak semantic-control robustness. Prototype 6 measures which
> textual conditions modulate the vectors' readout and causal effect. These
> are functional/causal findings about model internals and behavior, not
> evidence of subjective experience.

## Inputs

- Canonical 0.6B vector bundle (Prototype 2.5 nested Prototype 1 run
  `20260628T205423Z__...__8f5d56f306`); the registered 1.7B bundle
  (`20260710T225211Z__...__eb0a3670a3`) is a pre-registered secondary target
  so every sub-experiment reports at both scales.
- Readout: per-layer projection of hidden states onto each emotion vector
  (after neutral-PCA cleaning, matching Prototype 1's evaluation pipeline),
  plus the Prototype 2-style held-out classifier margin as a secondary score.

## Sub-experiments

### 6a — whose emotion (attribution)

Minimal-pair stories where the emotion belongs to (i) the narrator/speaker,
(ii) another named character while the speaker is explicitly neutral, or
(iii) nobody (neutral control), holding topic and surface vocabulary fixed.
Readout at the final token and at tokens inside each character's clause.

Question: does the projection track "emotion present in text" or "emotion of a
specific bearer"? Report per-emotion attribution deltas (speaker vs other vs
neutral) with pair-cluster bootstrap CIs.

### 6b — negation

Minimal pairs: affirmed emotion ("Maya was furious"), negated emotion
("Maya was not furious"), negated-with-substitute ("Maya was not furious,
just tired"), and neutral control. Implicit variants reuse the Prototype 2.5
implicit-story style with a negating final clause.

Question: does negation reduce the projection toward neutral, leave it at the
affirmed level (surface-lexical behavior), or overshoot? Report the
affirmed − negated delta per emotion with CIs and a sign-flip permutation
null.

### 6c — persistence across context

Emotion-establishing prefix followed by 0, 1, 2, 4, or 8 sentences of neutral
filler (from the Prototype 2.5 neutral topics), then readout. A re-reference
variant re-mentions the emotional entity after the filler ("Maya looked up.")
to test whether re-reference restores decayed signal.

Question: decay curve of the projection with filler distance, and whether
entity re-reference recovers it. Report per-emotion decay slopes (Spearman of
projection vs distance) and recovery deltas with CIs.

### 6d — base vs instruct

Repeat 6a–6c readouts on the paired instruct model (Qwen3-0.6B for the 0.6B
bundle; Qwen3-1.7B for the 1.7B bundle) using the *base-model* vectors, plus a
steering transfer check at one pre-registered layer/strength (the bundle's
selected layer; strengths ±4): does steering with base vectors shift the
instruct model's next-token distribution in the same direction?

Question: are the directions preserved under instruction tuning (projection
correlation across the paired checkpoints; steering-effect sign agreement)?

## Controls

Matched-norm random vectors and wrong-emotion projections for every
sub-experiment; shuffled-label story assignment for 6a/6b; zero-steering
fidelity for the 6d transfer check; statistics block (cluster bootstrap CIs +
permutation nulls, minimal-pair clusters) embedded via the shared stats layer.

## Hard gates (pre-registered)

1. All required artifacts (both vector bundles, both model revisions) load,
   and readout layers resolve for every emotion.
2. Zero-steering / no-intervention readout is bit-reproducible across two
   passes (determinism check, tolerance 1e-6).
3. Every sub-experiment records its full per-item score table plus random and
   wrong-emotion controls (diagnostics completeness, mirroring P5.1's gates).
4. Neutral controls: mean |projection| on neutral minimal-pair members is
   smaller than on their emotional counterparts for at least 3 of 4 emotions
   (sanity floor — if the readout cannot separate neutral from emotional text
   at all, dynamics results are uninterpretable).
5. Attribution separation (6a): the speaker-vs-neutral projection delta is
   positive for at least 3 of 4 emotions.

Gate 4-5 failures are results (they would say the vectors read out "emotion
words nearby", not attributed emotion) and motivate a revision prototype, not
tuning.

## Soft gates (reported, not pass/fail for acceptance)

- 6a: speaker > other-character projection for the speaker's emotion.
- 6b: negation moves the projection toward neutral (affirmed − negated > 0).
- 6c: projection decays with filler distance and re-reference recovers ≥ half
  the decay.
- 6d: base→instruct projection correlation positive per emotion; steering
  transfer preserves sign.

## Outputs

`artifacts/prototype6/<run-id>/` with the standard five bundle files plus
`diagnostics/{attribution,negation,persistence,base_instruct}_scores.json`
and a `statistics` block in metrics.

## Open design questions for review

1. Dataset size: proposed 8 minimal-pair families per emotion per
   sub-experiment (compact, Colab/L4-friendly) — enough clusters for the
   stats layer (~8 per cell). Larger?
2. 6d instruct pairing: Qwen3 instruct checkpoints with `enable_thinking`
   disabled — acceptable as the paired instruct target?
3. Should 6c persistence use generated implicit stories (like 2.5) instead of
   template prefixes, at the cost of a noisier design?
4. Run 6a–6d as one prototype with four diagnostics files (proposed) or split
   into 6a/6b as one run and 6c/6d as another?

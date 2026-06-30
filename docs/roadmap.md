# Replication roadmap

## Prototype 0 — activation capture and intervention

Prove the experimental apparatus on a small open-weight model. A paired-prompt
direction is used only to validate capture, vector construction, intervention,
strength scaling, metrics, and artifact recording.

## Prototype 1 — emotion-vector extraction

Generate implicit-emotion stories on held-out topic splits, compute per-layer
difference-in-means vectors, remove neutral-corpus principal components, and
validate on held-out stories.

Implemented as `fe-prototype1`. The committed configuration is a compact
feasibility run; the next milestone is accepting and registering its first run
before scaling the emotion and topic sets.

## Prototype 2 — semantic validation

Test implicit scenarios, numerical intensity sweeps, lexical controls, logit-lens
effects, shuffled-label controls, and cross-topic generalization.

Implemented as `fe-prototype2`. This prototype treats Prototype 1 as a completed
failed-but-informative feasibility run and asks whether its signal survives
semantic controls before moving to emotion-space geometry.

## Prototype 2.5 — revised extraction bridge

When Prototype 2 finds fragile semantic robustness, revise the compact
extraction run before geometry: expand topics and stories per topic/emotion,
balance angry-versus-afraid contexts, then rerun Prototype 2 controls on the new
vectors.

Implemented as `fe-prototype25`.

## Prototype 3 — emotion-space geometry

Measure cosine structure, clustering, PCA/UMAP, valence/arousal alignment, and
representational similarity across layers.

Implemented as `fe-prototype3`. This prototype consumes the accepted Prototype
2.5 handoff vectors and frames the result as diagnostic geometry before causal
steering.

## Prototype 4 — causal emotion steering

Run matching-token and free-generation interventions with dose-response,
specificity, random-vector, KL, and fluency controls.

## Prototype 5 — activity preferences

Reproduce pairwise preferences, Elo ratings, activation/preference correlations,
and causal Elo shifts under token-local steering.

## Prototype 6 — representation dynamics

Study local versus planned emotion, present versus other speakers, negation,
entity re-reference, and base-versus-instruct changes.

## Prototype 7 — alignment evaluations

In isolated simulations, evaluate reward hacking, sycophancy/harshness, and
blackmail-like behavior under emotion-vector interventions.

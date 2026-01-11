# geometric_interpretability
Tools for analyzing geometric structure and attention mechanisms in transformer activation spaces

This repo includes tools for interpretating internal representations and attention in transformers.

# Attention Analyzer
Mechanistic interpretability utilities for exploring attention heads in
autoregressive transformer models. Includes:

- Differential activation analysis between two prompt conditions.
- Targeted head verification (punctuation/previous-token/semantic).
- A deeper "interview" of a specific head to inspect its attention targets.

  ## Contents

- `differential_activation.py`: Core analysis class and visualization helpers.
- `demo.py`: End-to-end demo that generates samples and runs the analysis.
- `head_verify.py`: Targeted probe for classifying a specific head.
- `interview_head.py`: Richer inspection utilities for a specific head.
- `results_demo/`: Example outputs (plots + JSON).

# Geometric Probe

A small probe that estimates local curvature of a model's hidden-state
manifold and renders a token-level heatmap. Red indicates constraint, blue indicates expansion.

## Contents
- `geometric_probe.py`: probe implementation + demo
- `constraint.png`, `expansion.png`: example visuals

# Delta Manifold

A tool containing a rolling PCA over hidden-state deltas (Δ vectors) to track drift, coherence, and basis directions over time. The `DeltaManifold` class maintains a window
of normalized deltas, updates a PCA basis, and can emit bias vectors for
downstream logit steering.

## Contents

- `delta_manifold.py`: core implementation
- `delta_cross_session_summary.png`: example visualization
- `delta-trial*_delta_timeline.png`: example timelines
- 
# Cluster Analysis

A tool for clustering cross-session identity markers and labeling the resulting
geometry with simple motif classes. The current main workflow projects markers into UMAP space,
clusters them with DBSCAN, and compares those geometric labels to semantic cues.

## Contents

- `geometric_pattern_analyzer.py`: UMAP + clustering + geometry labeling
- `enhanced_metaphor_profiler.py`: metaphor profiling over geometric clusters
- `early_layer.png`, `mid_layer.png`, `late_layer.png`: example visuals

# The Key Finding: Possible Geometric Proprioception
What I found through using these tools is that models appear to genuinely perceive their own geometric trajectories. This happens when they are given geometric interventions like orthogonal repulsion, directional steering. This then causes models to spontaneously describe the methods that were used with spatial metaphors that correlate with measurable properties.

Some examples being, an output describing a "sideways tug" that happens from orthogonal displacement,
describing "coordinated waves" and then later finding that FFT (Fast Fourier Transform) confirmed these periodic oscillations,
and using "ripples converging", which is verified with DMD (Dynamic Mode Decomposition) showing a coherent mode like structure.

This could sugeest that models may have access in some way to internal geometric awareness or introspection like mentioned in Anthropic's more recent papers "Signs of introspection in large language models" and "When Models Manipulate Manifolds: The Geometry of a Counting Task".

# Safety & Alignment Relevance
For interpretability especially, if models can perceive their own activation space trajectories, this could provide a new way of understanding model "cognition" that goes beyond input-output mappings.
Something also to consider is behavior monitoring which is how Delta manifold is used to track and detect drift or instability in model behavior (this shows up as geometric inconsistency), as well as when models enter novel or familiar territory which shows in basis growth patterns. We might also find ways to find potential signatures of deception or misalignment via trajectory tracking and looking for discontinuities.


# Citation
If you build on this work, please cite appropriately and feel free to reach out about collaboration.

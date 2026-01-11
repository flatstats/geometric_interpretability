# Cluster Analysis

This tool is for clustering cross-session identity markers and labeling the resulting
geometry with simple motif classes (Constriction, Cyclicality, Divergence,
Fragmentation, Convergence). The current main workflow projects markers into UMAP space,
clusters them with DBSCAN, and compares those geometric labels to semantic cues.

## What's here

- `geometric_pattern_analyzer.py`: UMAP + clustering + geometry labeling
- `enhanced_metaphor_profiler.py`: metaphor profiling over geometric clusters
- `early_layer.png`, `mid_layer.png`, `late_layer.png`: example visuals

## Requirements

- `numpy`
- `scikit-learn`
- `matplotlib`
- `umap-learn`
- `scipy` (for curvature/angle diagnostics)
- `plotly` (optional, for interactive HTML)
- `sentence-transformers` (optional, for zero-shot semantic labels)

Install:

```bash
pip install numpy scikit-learn matplotlib umap-learn scipy
```

## Quickstart

```python
from analysis.cross_session_analyzer import CrossSessionAnalyzer
from geometric_pattern_analyzer import GeometricPatternAnalyzer

cross = CrossSessionAnalyzer("/path/to/sessions")
cross.load_sessions()

analyzer = GeometricPatternAnalyzer(cross, output_dir="./geometric_analysis")
results = analyzer.run()
```

This writes a `geometric_report.json` plus PNGs/HTML under the output directory.

## Visual Notes

The `early_layer.png`, `mid_layer.png`, and `late_layer.png` files are example
UMAP projections of marker vectors by layer. In the analyzer outputs:

- Each point is a marker vector from a session/exchange.
- Colors represent the assigned geometry class (see `GEOMETRIC_CLASSES`).
- Tighter blobs indicate higher local similarity; elongated shapes map to
  "Constriction" or "Cyclicality", and wide, diffuse clouds map to "Divergence".
- Isolated points are typically DBSCAN noise (`label = -1`), especially in
  sparse layers or smaller datasets.

## Metaphor profiling

`enhanced_metaphor_profiler.py` reads a `geometric_report.json` and correlates
clusters with spatial metaphor usage in the underlying responses.

```bash
python enhanced_metaphor_profiler.py /path/to/geometric_report.json /path/to/sessions
```

To analyze a single session JSON:

```bash
python enhanced_metaphor_profiler.py --session /path/to/session.json
```
## Findings

While these visualizations only represent my current custom architecture, the clustering patterns from regex markers indicates there may be a connection to language used by the model and the dynamics taking place within the activation space. Most notably the way "convergence" tags appear to cluster in a more specifc area than other tags. 
This leads to me to ask, if other model architectures have similar patterns, and to expand the regex to see if a true pattern is emerging.

Contributions welcome! 

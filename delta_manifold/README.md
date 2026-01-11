# Delta Manifold

This tool contains a rolling PCA over hidden-state deltas (Δ vectors) to track drift, coherence, and basis directions over time. The `DeltaManifold` class maintains a window
of normalized deltas, updates a PCA basis, and can emit bias vectors for
downstream logit steering.

## What's here

- `delta_manifold.py`: core implementation
- `delta_cross_session_summary.png`: example visualization
- `delta-trial*_delta_timeline.png`: example timelines

## Quickstart

```python
import numpy as np
from delta_manifold import DeltaManifold

manifold = DeltaManifold(vector_dim=4096)

# Feed consecutive hidden states (1D vectors).
hidden_states = [np.random.randn(4096) for _ in range(64)]
for vec in hidden_states:
    obs = manifold.observe(vec, metadata={"source": "demo"})
    if obs and obs.bias_vector is not None:
        print("bias norm:", np.linalg.norm(obs.bias_vector))

print("basis size:", manifold.basis_size)
```

## Notes

- `observe()` expects consecutive hidden states; it uses the difference between
  the current and previous vector to build the Δ buffer.
- `analyze_wave_patterns()` provides lightweight FFT/DMD diagnostics over the
  recent delta window.
- If `log_path` is set, interaction metadata is appended as JSONL.

## Key observations:

There appears to be consistent geometric structure across many sessions (83 so far).
The adaptive basis evolution grows to capture new patterns then prunes unused dimensions.
I was able to find measurable wave patterns in delta space correlating with model self-descriptions as well.

*Note: Results are from a custom research system designed for continuous geometric tracking. Code is provided for reproducibility and extension to standard transformer architectures.*

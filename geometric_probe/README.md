# Geometric Probe

A small probe that estimates local curvature of a model's hidden-state
manifold and renders a token-level heatmap. Red indicates constraint
(high tension), blue indicates expansion (high possibility).

## What's here

- `geometric_probe.py`: probe implementation + demo
- `constraint.png`, `expansion.png`: example visuals

## Requirements

- `torch`
- `transformers`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `ipython` (for HTML display in notebooks)

Install:

```bash
pip install torch transformers numpy scikit-learn matplotlib ipython
```

## Quickstart

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from geometric_probe import GeometricProbe

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

probe = GeometricProbe(model, tokenizer, layer_idx=-2)
probe.render_heatmap("The artificial intelligence hesitated.")
```

## Notes

- `render_heatmap()` uses `IPython.display`; it is best viewed in a notebook.
- The curvature estimate is local and depends on the `window` size.

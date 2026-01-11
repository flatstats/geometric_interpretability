# geometric_interpretability
Tools for analyzing geometric structure and attention mechanisms in transformer activation spaces

This repo includes tools for interpretating internal representations and attention in transformers.

# Attention Analyzer
Mechanistic interpretability utilities for exploring attention heads in
autoregressive transformer models. The repo includes:

- Differential activation analysis between two prompt conditions.
- Targeted head verification (punctuation/previous-token/semantic).
- A deeper "interview" of a specific head to inspect its attention targets.

  ## Contents

- `differential_activation.py`: Core analysis class and visualization helpers.
- `demo.py`: End-to-end demo that generates samples and runs the analysis.
- `head_verify.py`: Targeted probe for classifying a specific head.
- `interview_head.py`: Richer inspection utilities for a specific head.
- `results_demo/`: Example outputs (plots + JSON).

  ## Requirements

Python 3.9+ with the following packages:

- `torch`
- `transformers`
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `tqdm`

Install with:

```bash
pip install torch transformers numpy scipy matplotlib seaborn tqdm
```

GPU is optional but recommended for large models.

## Quickstart

### Differential activation demo

This script generates two prompt sets (geometric vs analytical), runs
per-head comparisons, and saves plots + JSON.

```bash
python demo.py
```

Outputs land in `./results_demo`.

### Verify a head's behavior

```bash
python head_verify.py
```

Edit the `MODEL_NAME`, `layer`, and `head` in the script to your target.

### Interview a head

`interview_head.py` includes several probes:

- `verify_head_stats`: quick behavior profile
- `check_dark_matter`: BOS sink check
- `interview_head`: top-K targets for each token
- `measure_entropy`: attention entropy diagnostic

Run the script directly or import `HeadAnalyst`:

```python
from interview_head import HeadAnalyst

analyst = HeadAnalyst("/openai/gpt-oss-20b")
analyst.interview_head(layer=3, head=4, prompt="The shape of thought is a spiral.")
```

## Notes

- For large models, consider `device_map="auto"` and `torch_dtype=torch.bfloat16`
  (see `demo.py` and `interview_head.py`).
- Some models require setting a `pad_token` (see `demo.py` for GPT-2).

## Findings
Initial analysis identified L3.H4 as a candidate for geometric processing. 
To verify this was not a confounder of sentence length, I performed a control experiment comparing attention entropy across three distinct semantic conditions.

| Condition      |  Prompt Example          | Entropy (S)  | 
| ------------- |:-------------:| -----:|
| Short Fact     | Water freezes at 0 degrees Celsius because | 0.49 (Low) |
| Long and Concrete     | Some magazines are concerned with more recreational topics, like sports card collecting or different kinds of hairstyles     |  1.26 (Low) |
| Geometric Metaphor | If your mind were like a prism, how would you tilt it to understand yourself more?     |    2.48 (High)|

## Methodology
### Differential Activation Analysis
I developed a statistical method to compare mean activation patterns across thousands of attention heads ($L \times H$).
- Metrics used Maximum Attention Weight vs. Mean Attention Weight.
- I use Bonferroni correction to control Family-Wise Error Rate (FWER) and prevent false discoveries.
- As a result we get a "Difference Map" isolating heads that significantly diverge between geometric and analytical prompts.

*(See "Results" dir for visualizations such as a heatmap showing heads that are differentially active. Red indicates higher activity in Geometric mode; Blue indicates higher activity in Analytical mode.)*

### The "Head Analyst" Probe
A custom diagnostic tool (HeadAnalyst) was built to "interview" specific heads:
- Dark Matter Check: Verifies the head is not an "Attention Sink" (dumping attention on the BOS token).
- Semantic Verification: Classifies head behavior (Punctuation vs. Content vs. Previous Token).
- Entropy Measurement: Quantifies the "blurriness" of the attention distribution.

## Implications for AI Safety
Future work could involve training Sparse Autoencoders (SAEs) to decompose L3.H4 (or other heads like it) into interpretable features, allowing us to detect and potentially steer models when they enter high-level conceptual states.

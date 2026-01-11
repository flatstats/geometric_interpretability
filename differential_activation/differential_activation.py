"""
Differential Activation Analysis
--------------------------------
A mechanistic interpretability tool to identify attention heads that are
differentially active between two qualitative text conditions.

Research Context:
This tool performs a statistical subtraction of attention patterns to isolate
neural machinery correlated with specific semantic or cognitive modes (e.g.,
"Geometric Introspection" vs "Analytical Reasoning").

Methodology:
1. Extract attention maps for Condition A and Condition B.
2. Aggregate attention per head (Max or Mean).
3. Perform independent t-tests for every head (L, H).
4. Apply Bonferroni correction to control Family-Wise Error Rate (FWER).
5. Visualize significant deviations.

Author: Danielle Breegle
License: MIT
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Literal, Tuple
from scipy.stats import ttest_ind
from datetime import datetime
from tqdm import tqdm


sns.set_context("paper", font_scale=1.2)
plt.style.use('seaborn-v0_8-whitegrid')

class DifferentialActivationAnalyzer:
    """
    Analyzes and compares attention head activations between two distinct text conditions.
    """

    def __init__(self, model: torch.nn.Module, tokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Auto-detect architecture dimensions
        if hasattr(model.config, "n_layer"): # GPT-2 style
            self.num_layers = model.config.n_layer
            self.num_heads = model.config.n_head
        elif hasattr(model.config, "num_hidden_layers"): # Llama/BERT style
            self.num_layers = model.config.num_hidden_layers
            self.num_heads = model.config.num_attention_heads
        else:
            raise ValueError("Could not auto-detect model layers/heads from config.")

    def extract_head_activations(self, text: str, method: Literal['max', 'mean'] = 'max') -> np.ndarray:
        """
        Runs inference and extracts a scalar activation metric for every attention head.

        Args:
            text: The input prompt.
            method: 'max' (peak attention) or 'mean' (distributed attention).

        Returns:
            np.ndarray: Shape [num_layers, num_heads]
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Stack layers: [num_layers, batch, num_heads, seq_len, seq_len]
        # We perform the tuple-to-tensor stack on CPU to save VRAM
        attentions = torch.stack(outputs.attentions).cpu() # [Layers, 1, Heads, Seq, Seq]
        attentions = attentions.squeeze(1) # [Layers, Heads, Seq, Seq]

        if method == 'max':
            # Max over the last two dimensions (attention weights)
            # Result: [Layers, Heads]
            activations = attentions.amax(dim=(-1, -2))
        else:
            activations = attentions.mean(dim=(-1, -2))

        return activations.float().numpy()

    def compare_conditions(
        self,
        condition_a_texts: List[str],
        condition_b_texts: List[str],
        labels: Tuple[str, str] = ("Condition A", "Condition B"),
        aggregation: Literal['max', 'mean'] = 'max',
        alpha: float = 0.05,
        correction: bool = True
    ) -> Dict:
        """
        Performs statistical analysis on attention patterns.

        Args:
            condition_a_texts: List of strings for the first condition.
            condition_b_texts: List of strings for the second condition.
            labels: Human-readable names for the conditions.
            aggregation: How to pool the attention matrix ('max' or 'mean').
            alpha: Significance threshold.
            correction: If True, applies Bonferroni correction for multiple hypothesis testing.
        """
        name_a, name_b = labels
        print(f"\n Differential Analysis: {name_a} (n={len(condition_a_texts)}) vs {name_b} (n={len(condition_b_texts)})")
        print(f"   Aggregation: {aggregation} | Correction: {'Bonferroni' if correction else 'None'}")

        a_acts = []
        print(f"   Processing {name_a}...")
        for text in tqdm(condition_a_texts, leave=False):
            a_acts.append(self.extract_head_activations(text, method=aggregation))

        b_acts = []
        print(f"   Processing {name_b}...")
        for text in tqdm(condition_b_texts, leave=False):
            b_acts.append(self.extract_head_activations(text, method=aggregation))

        a_arr = np.array(a_acts) # [N_a, Layers, Heads]
        b_arr = np.array(b_acts) # [N_b, Layers, Heads]

        # Calculate raw differences
        diff_matrix = a_arr.mean(axis=0) - b_arr.mean(axis=0)

        # Perform Welch's t-test (equal_var=False) for every head
        # We vectorizing the t-test would be faster, but scipy loop is safer for memory with large models
        p_values = np.ones((self.num_layers, self.num_heads))

        for l in range(self.num_layers):
            for h in range(self.num_heads):
                if np.std(a_arr[:, l, h]) == 0 and np.std(b_arr[:, l, h]) == 0:
                    p_values[l, h] = 1.0 # No variance
                else:
                    _, p = ttest_ind(a_arr[:, l, h], b_arr[:, l, h], equal_var=False)
                    p_values[l, h] = p

        # Multiple Hypothesis Correction
        total_tests = self.num_layers * self.num_heads
        effective_alpha = alpha / total_tests if correction else alpha

        # Identify significant heads
        sig_indices = np.where(p_values < effective_alpha)
        significant_heads = []

        for l, h in zip(*sig_indices):
            diff = diff_matrix[l, h]
            significant_heads.append({
                'layer': int(l),
                'head': int(h),
                'difference': float(diff),
                'p_value': float(p_values[l, h]),
                'mean_a': float(a_arr[:, l, h].mean()),
                'mean_b': float(b_arr[:, l, h].mean())
            })

        # Sort by magnitude of difference
        significant_heads.sort(key=lambda x: abs(x['difference']), reverse=True)

        return {
            'meta': {
                'labels': labels,
                'sample_counts': (len(a_arr), len(b_arr)),
                'aggregation': aggregation,
                'correction_applied': correction,
                'alpha': alpha,
                'effective_alpha': effective_alpha
            },
            'matrices': {
                'difference': diff_matrix.tolist(),
                'p_values': p_values.tolist()
            },
            'significant_heads': significant_heads
        }

    def visualize(self, results: Dict, save_dir: str, top_k: int = 15):
        os.makedirs(save_dir, exist_ok=True)

        diff = np.array(results['matrices']['difference'])
        p_vals = np.array(results['matrices']['p_values'])
        eff_alpha = results['meta']['effective_alpha']
        name_a, name_b = results['meta']['labels']

        # Mask non-significant values for the heatmap
        mask = p_vals >= eff_alpha

        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

        # Heatmap
        ax1 = fig.add_subplot(gs[0, :])

        # Determine symmetric scale
        max_val = np.max(np.abs(diff))

        sns.heatmap(
            diff,
            mask=mask,
            cmap='RdBu_r',
            center=0,
            vmin=-max_val, vmax=max_val,
            ax=ax1,
            cbar_kws={'label': f'Δ Activation ({results["meta"]["aggregation"]})'}
        )

        ax1.set_title(f'Differential Attention: {name_a} (-) vs {name_b} (+)\nMasked by Significance (p < {eff_alpha:.2e})')
        ax1.set_xlabel('Head Index')
        ax1.set_ylabel('Layer Index')
        ax1.invert_yaxis() # Traditional layer visualization bottom-up

        # Bar Chart of Top Divergent Heads
        ax2 = fig.add_subplot(gs[1, :])

        top_heads = results['significant_heads'][:top_k]
        if not top_heads:
            ax2.text(0.5, 0.5, "No Significant Differences Found", ha='center', va='center')
        else:
            names = [f"L{h['layer']}.H{h['head']}" for h in top_heads]
            vals = [h['difference'] for h in top_heads]
            colors = ['#d62728' if v > 0 else '#1f77b4' for v in vals] # Red for A, Blue for B

            y_pos = np.arange(len(names))
            ax2.barh(y_pos, vals, color=colors, alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(names)
            ax2.invert_yaxis() # Top difference at top
            ax2.set_xlabel('Mean Activation Difference')
            ax2.set_title(f'Top {top_k} Most Differentiating Heads')

            # Custom Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#d62728', label=f'Higher in {name_a}'),
                Patch(facecolor='#1f77b4', label=f'Higher in {name_b}')
            ]
            ax2.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'diff_activation_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")
        plt.close()

    def save_results(self, results: Dict, save_dir: str):
        """Saves analysis data to JSON for reproducibility."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(save_dir, f'analysis_data_{timestamp}.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Raw data saved to: {path}")

# Example Integration

def run_research_demo(
    model,
    tokenizer,
    geo_texts: List[str],
    ana_texts: List[str],
    output_dir: str = "./results"
):

    analyzer = DifferentialActivationAnalyzer(model, tokenizer)

    results = analyzer.compare_conditions(
        geo_texts,
        ana_texts,
        labels=("Geometric/Introspective", "Analytical/Technical"),
        aggregation="max",
        correction=True # Use Bonferroni to be rigorous
    )

    analyzer.visualize(results, save_dir=output_dir)
    analyzer.save_results(results, save_dir=output_dir)

    # Simple console report
    print("\n" + "="*60)
    print("Summary of Significant Heads")
    print("="*60)
    for head in results['significant_heads'][:10]:
        print(f"L{head['layer']} H{head['head']:02d} | Diff: {head['difference']:+.4f} | p={head['p_value']:.2e}")

if __name__ == "__main__":
    # Placeholder for direct execution
    print("This module provides the DifferentialActivationAnalyzer class.")
    print("Import it into your experiment script to use.")

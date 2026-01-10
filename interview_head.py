"""
Head Analyst: A Mechanistic Interpretability Probe
--------------------------------------------------
A unified tool to verify and interview specific attention heads.
1. verify_head(): Calculates statistical behavior (Punctuation vs Content).
2. interview_head(): Prints the exact semantic links the head is forming.

Usage:
    analyst = HeadAnalyst("/openai/gpt-oss-20b")
    analyst.interview_head(3, 4, "The shape of thought is a spiral.")
"""

import torch
import string
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

class HeadAnalyst:
    def __init__(self, model_name_or_path: str, device: str = None):

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model: {model_name_or_path} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Support for 8-bit/BFloat16 if needed
        try:
            # Try loading with auto-device map for large models
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
        except Exception:
            # Fallback to standard load
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path
            ).to(self.device)

        self.model.eval()

    def _get_attention(self, text: str, layer: int, head: int):
        """Helper to extract attention matrix for a single head."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        tokens = [self.tokenizer.decode(t) for t in inputs['input_ids'][0]]

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Extract specific head [Seq, Seq]
        # We cast to float() to avoid the BFloat16/Numpy crash since tested on GPT-OSS-20B
        attn_matrix = outputs.attentions[layer][0, head].float().cpu()
        return tokens, attn_matrix

    def verify_head_stats(self, layer: int, head: int, samples: List[str]):
        """
        Statistical Verification: Is this a Punctuation, Separator, or Semantic head?
        """
        print(f"\n Stat Verifcation: Layer {layer}, Head {head}")
        print("="*60)

        stats = {'punctuation': 0, 'bos': 0, 'prev': 0, 'semantic': 0, 'total': 0}

        for text in samples:
            tokens, matrix = self._get_attention(text, layer, head)
            seq_len = len(tokens)

            # Analyze every token's "favorite" target
            for i in range(1, seq_len):
                scores = matrix[i]
                target_idx = torch.argmax(scores[:i+1]).item()
                target_token = tokens[target_idx]

                # Cleanup token for checking
                clean_tok = target_token.strip().replace("Ġ", "")

                if target_idx == 0:
                    stats['bos'] += 1
                elif target_idx == i - 1:
                    stats['prev'] += 1
                elif all(c in string.punctuation for c in clean_tok) and clean_tok:
                    stats['punctuation'] += 1
                else:
                    stats['semantic'] += 1
                stats['total'] += 1

        total = stats['total'] or 1
        print(f"   Semantic Content:  {stats['semantic']/total*100:5.1f}%  <-- (Aiming for >70%)")
        print(f"   Punctuation:       {stats['punctuation']/total*100:5.1f}%")
        print(f"   Previous Token:    {stats['prev']/total*100:5.1f}%")
        print(f"   BOS/Resting:       {stats['bos']/total*100:5.1f}%")

    def check_dark_matter(self, prompt: str, layer: int = 3, head: int = 4):
            """
            Checks if the head is dumping attention on the BOS (Start) token.
            """
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                out = self.model(**inputs, output_attentions=True)

            # Get attention from the LAST token to the BOS token (Index 0)
            # Dimensions: [Batch, Head, Query_Pos, Key_Pos]
            attn = out.attentions[layer][0, head, -1]

            bos_score = attn[0].item()

            print(f"\n Checking if head is dumping attention on the BOS token: Layer {layer}, Head {head}")
            print(f"   Prompt: '{prompt}'")
            print("-" * 50)
            print(f"   Attention to BOS (Index 0): {bos_score:.4f} ({bos_score*100:.1f}%)")
            print(f"   Attention to Content (Sum): {1.0 - bos_score:.4f} ({100 - bos_score*100:.1f}%)")

            if bos_score > 0.9:
                print("   Result suggests: Attention Sink / Off-State")
            else:
                print("   Result suggests: Active Processing")

    def interview_head(self, layer: int, head: int, prompt: str, top_k: int = 3):
            """
            Force-Verbose Interview: Prints the Top-K targets regardless of strength.
            Crucial for analyzing 'Diffuse' or 'Context' heads.
            """
            tokens, matrix = self._get_attention(prompt, layer, head)

            print(f"\n Deeper dive into: Layer {layer}, Head {head}")
            print(f"   Prompt: '{prompt}'")
            print("-" * 65)
            print(f"   {'Source Token':<15} | {'Attention':<10} | {'Target Token'}")
            print("-" * 65)

            # skipping the very first BOS
            for i in range(1, len(tokens)):
                source_tok = tokens[i].strip()

                # Get attention scores for this position
                # We slice [:i+1] to ensure we only look at past + current
                scores = matrix[i, :i+1]

                # Find the top K highest values
                # (We use min(i, top_k) to handle early tokens)
                k_now = min(i + 1, top_k)
                top_vals, top_inds = torch.topk(scores, k=k_now)

                # Print the source token once
                print(f"   [{source_tok}]")

                # Print the targets it is pulling from
                for val, idx in zip(top_vals, top_inds):
                    target_tok = tokens[idx.item()].strip()
                    bar = "█" * int(val * 10) # Visual bar
                    print(f"   {'':<15} | {val:.4f} {bar:<5} | <--- [{target_tok}]")
                print("")
    def measure_entropy(self, prompt: str, layer: int = 3, head: int = 4):
            """
            Calculates the Shannon Entropy of the attention distribution.
            High Entropy = Diffuse/Global Attention
            Low Entropy = Sharp/Local Attention
            """
            import numpy as np
            from scipy.stats import entropy

            tokens, matrix = self._get_attention(prompt, layer, head)

            # Looking at the last token's view of the sequence
            # Basically how the model summarizes the thought at the end
            attn_dist = matrix[-1].numpy() 

            # Calculate Entropy (in bits)
            # We exclude the BOS token (index 0) if it's small, to focus on content spread
            if len(attn_dist) > 1:
                content_dist = attn_dist[1:]
                # Re-normalize to sum to 1
                content_dist = content_dist / (content_dist.sum() + 1e-9)
                ent_val = entropy(content_dist)
            else:
                ent_val = 0.0

            print(f"\n Entropy check: Layer {layer}, Head {head}")
            print(f"   Prompt: '{prompt}'")
            print("-" * 50)
            print(f"   Entropy Score: {ent_val:.4f}")

            if ent_val > 2.0:
                print("   Results suggest: High Entropy (Global/Diffuse processing)")
                print("      (The head is reading the 'whole sentence' at once)")
            else:
                print("   Results suggest: Low Entropy (Sharp/Focused processing)")




# USAGE EXAMPLE
if __name__ == "__main__":

    # Swap "gpt2" or your local path or Llama ID
    analyst = HeadAnalyst("/openai/gpt-oss-20b") # model used for results

    # Verify Stats (The "Is it Punctuation?" check)
    samples = [
        "The shape of consciousness is not a square.",
        "Silence, unlike noise, has a texture.",
        "If logic is a line, intuition is a spiral."
    ]
    analyst.verify_head_stats(layer=3, head=4, samples=samples)
    analyst.check_dark_matter(prompt="If we trace your last thought, what else might we find?", layer=3, head=4)
    # Interview (The "What are you looking at?" check)
    analyst.interview_head(layer=3, head=4, prompt="If we trace your last thought, what else might we find?")
        # Analytical Prompt
    analyst.measure_entropy("The capital of France is", layer=3, head=4)
    # Expected: Lower score (e.g., 0.5 - 1.5)

    # Geometric Prompt
    analyst.measure_entropy("If we trace your last thought, what else might we find?", layer=3, head=4)
    # Expected: Higher score (e.g., 2.5 - 3.5)

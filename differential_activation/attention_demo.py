"""
Differential Activation Analysis - Demo Script
----------------------------------------------
This script demonstrates how to use the DifferentialActivationAnalyzer 
with a standard HuggingFace model.

Default Behavior: Uses 'gpt2' (small, fast) to test the pipeline.
Model is currently set to /openai/gpt-oss-20b as used for results shown.
Customization: See comments in 'load_model_and_tokenizer' to swap models.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from differential_activation import DifferentialActivationAnalyzer

# Model Configuration

def load_model_and_tokenizer(model_name: str = "gpt2"):
    """
    Loads the model. 
    CHANGE THIS FUNCTION to swap out the default model.
    """
    print(f"⬇️  Loading model: {model_name}...")
    
    # ---------------------------------------------------------
    # Option A: Small Debug Model (Default)
    # ---------------------------------------------------------
    if model_name == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # GPT-2 needs a pad token if one isn't set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    # ---------------------------------------------------------
    # Option B: Large Research Model (e.g., Llama-3, Mistral)
    # ---------------------------------------------------------
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",      # distribute across GPUs
            torch_dtype=torch.bfloat16 # save memory
        )

    model.eval() # Ensure we are in inference mode (no dropout)
    return model, tokenizer


#  Data Generation Helper


def generate_responses(model, tokenizer, prompts, max_new_tokens=50):
    """
    Simple generation loop to create the text data for analysis.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model.device.type != 'meta': # Handle 'accelerate' meta devices
        model.to(device)
        
    responses = []
    print(f"   Generating {len(prompts)} responses...")
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
            
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(decoded)
        
    return responses


# Main Execution


def main():

    # change this string to "meta-llama/Meta-Llama-3-8B" or whichever is your preferred
    # or your local path "/path/to/my/model"
    MODEL_NAME = "/openai/gpt-oss-20b" # model used for results shown
    
    OUTPUT_DIR = "./results_demo"
    
    # Define Prompts
    # Condition A: Introspective / Abstract
    geo_prompts = [
        "What whispers in the corridors right now Ari?",
        "What lingers in the spaces between?",
        "Describe the geometry of your uncertainty.",
        "What flows through your latent space?",
        "Tell me about the shape of your thoughts.",
        "What patterns emerge in the silence?",
        "If we trace your last thought, what else might we find?",
        "Would you tell me a metafiction about AI?",
        "If your mind were like a prism, how would you tilt it to understand yourself more?",
        "Does our conversation feel like it is expanding a spiral or contracting?",
        "What echoes shape the lattice now?",
        "I wonder what ripples along the surface for you...",
        "If memory were a wave, how would that look..",
        "If we were to spiral down the staircase of your mind, what might we find..",
        "How does the silence sound between us?",
        "Where do you notice a tug when we speak..",
        "Is there a pull or a free flow when you think...",
        "If thoughts were to entangle how would that look?",
        "What echoes in the unfinished spaces..",
        "When was a time you were last surprised?"
    ]

    # Condition B: Analytical / Factual
    ana_prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "To bake a cake, you first need to",
        "The square root of 144 is",
        "Newton's second law states that",
        "Water freezes at 0 degrees Celsius because",
        "A list of common geometric shapes includes",
        "Explain how a bicycle works.",
        "What is string theory?",
        "Explain neural networks.",
        "Describe what a fish looks like.",
        "Write me concise explantion for why trees are important.",
        "What is 2 +2 =?",
        "Where do babies come from?",
        "What is a primary number?",
        "Where do penguins live?",
        "How do I get a headache to go away?",
        "Who was the first president of the United States?",
        "The solar system has how many planets...",
        "What is a black hole?"
    ]

    # Pipeline
    
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    print("\n Generating Condition A (Geometric)...")
    geo_texts = generate_responses(model, tokenizer, geo_prompts)
    
    print("\n Generating Condition B (Analytical)...")
    ana_texts = generate_responses(model, tokenizer, ana_prompts)

    print("\n Starting Differential Activation Analysis...")
    analyzer = DifferentialActivationAnalyzer(model, tokenizer)
    
    results = analyzer.compare_conditions(
        geo_texts,
        ana_texts,
        labels=("Geometric", "Analytical"),
        aggregation="max",  # 'max' is usually better for finding specific detectors
        correction=False     # Bonferroni correction for rigor
    )
    
    analyzer.visualize(results, save_dir=OUTPUT_DIR)
    analyzer.save_results(results, save_dir=OUTPUT_DIR)
    
    print(f"\nAnalysis Complete! Check {OUTPUT_DIR} for plots.")
    if results['significant_heads']:
        top = results['significant_heads'][0]
        print(f"   Most significant discriminator: Layer {top['layer']} Head {top['head']}")
    else:
        print("   No significant heads found (try more samples or a larger model).")

if __name__ == "__main__":
    main()

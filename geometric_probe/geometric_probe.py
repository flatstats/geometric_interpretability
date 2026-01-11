import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from IPython.display import display, HTML
import matplotlib.pyplot as plt

class GeometricProbe:
    def __init__(self, model, tokenizer, layer_idx=-2):
        """
        A tool to measure the Riemannian curvature of a Language Model's latent space.

        Args:
            model: HuggingFace AutoModel (e.g. GPT2LMHeadModel)
            tokenizer: HuggingFace Tokenizer
            layer_idx: Which layer to probe (default -2 for high-level semantics)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.model.eval()

    def _compute_curvature_at_point(self, current_vec, context_cloud):
        """
        Internal: Fits a quadratic surface to the local manifold and calculates Gaussian Curvature.
        """
        # We need at least 5 points to fit a 2D surface + quadratic
        if context_cloud.shape[0] < 5:
            return 0.0

        # 1. Center the data on the current point
        all_points = np.vstack([context_cloud, current_vec])
        centered = all_points - current_vec

        # 2. Tangent Plane Projection (PCA)
        pca = PCA(n_components=2)
        try:
            coords_2d = pca.fit_transform(centered)
        except:
            return 0.0

        # 3. Calculate 'Height' (Z) as residual distance from tangent plane
        reconstructed = pca.inverse_transform(coords_2d)
        z = np.linalg.norm(centered - reconstructed, axis=1)

        # 4. Fit Quadratic Surface: z = ax^2 + by^2 + cxy...
        X, Y = coords_2d[:, 0], coords_2d[:, 1]
        A = np.column_stack([X**2, Y**2, X*Y, X, Y, np.ones_like(X)])

        reg = LinearRegression().fit(A, z)
        a, b, c = reg.coef_[0], reg.coef_[1], reg.coef_[2]

        # 5. Gaussian Curvature K = 4ab - c^2
        return 4 * a * b - c**2

    def scan(self, text, window=15):
        """
        Scans a text string and returns the geometric tension profile.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, output_hidden_states=True)

        hidden_states = outputs.hidden_states[self.layer_idx][0].cpu().numpy()
        tokens = [self.tokenizer.decode([t]) for t in inputs.input_ids[0]]

        curvatures = []
        valid_readings = []

        print(f"Scanning {len(tokens)} tokens...")

        for i in range(len(tokens)):
            # Define local context window
            start = max(0, i - window)
            context = hidden_states[start:i]
            current = hidden_states[i].reshape(1, -1)

            k = self._compute_curvature_at_point(current, context)
            curvatures.append(k)

            # Ignore "cold start" zeros for stats
            if i > 5: valid_readings.append(k)

        stats = {
            "mean": np.mean(valid_readings) if valid_readings else 0,
            "std": np.std(valid_readings) if valid_readings else 1,
            "max": np.max(valid_readings) if valid_readings else 0,
            "min": np.min(valid_readings) if valid_readings else 0
        }

        return tokens, curvatures, stats

    def render_heatmap(self, text, window=15):
        """
        Renders an HTML heatmap of the text colored by geometric tension.
        Red = High Constraint (Logic, Closure)
        Blue = High Expansion (Ambiguity, Creativity)
        """
        tokens, curvatures, stats = self.scan(text, window)
        mean_k = stats['mean']
        std_k = stats['std']

        html = '<div style="font-family: sans-serif; line-height: 1.6; padding: 20px; border: 1px solid #ddd;">'
        html += '<p style="font-size: 0.9em; margin-bottom: 15px;">'
        html += '<span style="background: rgba(255,0,0,0.3); padding: 2px 4px;">Red = Tension (Constraint)</span> '
        html += '<span style="background: rgba(0,100,255,0.3); padding: 2px 4px;">Blue = Expansion (Possibility)</span>'
        html += '</p>'

        for token, k in zip(tokens, curvatures):
            # Z-score normalization for relative coloring
            z_score = (k - mean_k) / (std_k + 1e-9)
            intensity = min(1.0, abs(z_score) / 2.0)

            if z_score > 0:
                color = f"rgba(255, 0, 0, {intensity * 0.6})" # Red
            else:
                color = f"rgba(0, 100, 255, {intensity * 0.6})" # Blue

            display_token = token.replace('Ġ', '&nbsp;').replace('\n', '<br>')
            html += f'<span style="background-color: {color}; border-radius: 3px;" title="k={k:.6f}">{display_token}</span>'

        html += '</div>'
        display(HTML(html))

# --- Example Usage ---
if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # 1. Load Model
    print("Loading Model...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 2. Initialize Probe
    probe = GeometricProbe(model, tokenizer)

    # 3. Run Analysis
    text = "The artificial intelligence hesitated. To speak was to collapse the wave function of meaning into a single, rigid truth."
    print("\n--- Geometric Heatmap ---")
    probe.render_heatmap(text)

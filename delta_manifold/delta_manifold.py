"""
core/delta_manifold.py

DeltaManifold manages Δ-space modelling for transformer hidden states. It
collects consecutive hidden-state differences, performs PCA to discover the
principal drift directions, and exposes helpers for drift diagnostics and
logit-bias construction.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np


@dataclass
class DeltaObservation:
    """Metadata returned after feeding a new hidden state into the manifold."""

    delta_norm: float
    residual: float
    projections: np.ndarray
    basis_size: int
    bias_vector: Optional[np.ndarray]
    coherence: float = 0.0  # 0.0 = Noise, 1.0 = Pure Tone/Lullaby
    metadata: Optional[Dict[str, Any]] = None


class DeltaManifold:
    """Maintains a rolling PCA over Δ vectors."""

    def __init__(
        self,
        vector_dim: int,
        window_size: int = 512,
        min_samples: int = 48,
        variance_threshold: float = 0.9,
        bias_scale: float = 0.075,
        prune_interval: int = 400,
        prune_usage_threshold: float = 0.1,
        log_path: Optional[str] = "logs/interaction_log.jsonl",
    ):
        self.vector_dim = vector_dim
        self.window_size = window_size
        self.min_samples = min_samples
        self.variance_threshold = variance_threshold
        self.bias_scale = bias_scale
        self.prune_interval = prune_interval
        self.prune_usage_threshold = prune_usage_threshold

        self._buffer: Deque[np.ndarray] = deque(maxlen=window_size)
        self._delta_energy: Deque[float] = deque(maxlen=window_size)
        self._components: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._total_variance: float = 0.0
        self._needs_recompute: bool = False
        self._last_vector: Optional[np.ndarray] = None
        self._basis_is_inherited: bool = False  # Track if basis came from disk
        self._observations_since_recompute: int = 0  # Track how long basis has been stable
        self._basis_usage: Optional[np.ndarray] = None  # Track which dimensions are actually used
        self._usage_decay: float = 0.95  # Decay factor for usage tracking
        self._since_last_prune: int = 0
        self._interaction_log: Deque[Dict[str, Any]] = deque(maxlen=window_size)
        self.log_path = log_path

        if self.log_path:
            try:
                os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            except OSError:
                # Non-fatal; just disable persistence if we cannot create the directory
                print(f"Δ warning: could not create log directory for {self.log_path}")
                self.log_path = None

    # ---------------------------------------------------------------- wave viz
    def analyze_wave_patterns(
        self,
        window: int = 128,
        fft_top_k: int = 5,
        dmd_rank: Optional[int] = 6
    ) -> Dict[str, Any]:
        """Compute lightweight FFT + DMD diagnostics over recent Δ vectors.

        Returns a dict with `available` flag plus FFT and DMD summaries that
        downstream callers can turn into visualisations.
        """
        sample_count = len(self._buffer)
        if sample_count < 4:
            return {
                "available": False,
                "reason": "Insufficient Δ samples",
                "sample_count": sample_count,
            }

        window_size = int(max(4, min(window, sample_count)))
        recent_deltas = np.stack(list(self._buffer)[-window_size:], axis=0).astype(np.float32)

        energy_series = np.asarray(list(self._delta_energy)[-window_size:], dtype=np.float32)
        if energy_series.shape[0] != window_size:
            # Align lengths defensively
            min_len = min(window_size, energy_series.shape[0], recent_deltas.shape[0])
            recent_deltas = recent_deltas[-min_len:]
            energy_series = energy_series[-min_len:]
            window_size = int(min_len)
        if window_size < 4:
            return {
                "available": False,
                "reason": "Insufficient aligned Δ samples",
                "sample_count": sample_count,
                "window_size": window_size,
            }

        # Weight deltas by their raw energy to capture true movement magnitude
        weighted = recent_deltas * energy_series.reshape(-1, 1)
        centered = weighted - weighted.mean(axis=0, keepdims=True)

        # --- FFT across the temporal axis
        fft_values = np.fft.fft(centered, axis=0)
        magnitudes = np.abs(fft_values)
        avg_magnitude = np.mean(magnitudes, axis=1)

        half = window_size // 2
        freqs = np.fft.fftfreq(window_size)[:half]
        positive_magnitude = avg_magnitude[:half]

        top_k = int(max(1, min(fft_top_k, len(positive_magnitude))))
        peak_indices = np.argsort(positive_magnitude)[-top_k:][::-1]
        peak_frequencies = [
            {
                "frequency": float(freqs[idx]),
                "magnitude": float(positive_magnitude[idx]),
                "index": int(idx),
            }
            for idx in peak_indices
        ]

        # --- DMD on the same window (low-rank to keep it light)
        dmd_modes: List[Dict[str, float]] = []
        try:
            snapshots = centered[:-1].T  # [dim, m-1]
            snapshots_next = centered[1:].T
            u, s, vh = np.linalg.svd(snapshots, full_matrices=False)

            r = min(len(s), snapshots.shape[0], snapshots.shape[1], dmd_rank or len(s))
            if r > 0:
                u_r = u[:, :r]
                s_r = s[:r]
                vh_r = vh[:r, :]

                inv_s = np.divide(1.0, s_r, out=np.zeros_like(s_r), where=s_r > 1e-8)
                a_tilde = (u_r.T @ snapshots_next @ vh_r.T) * inv_s.reshape(-1, 1)
                eigvals, eigvecs = np.linalg.eig(a_tilde)

                phi = snapshots_next @ vh_r.T @ np.diag(inv_s) @ eigvecs
                mode_energy = np.linalg.norm(phi, axis=0)
                angles = np.angle(eigvals)

                order = np.argsort(mode_energy)[::-1]
                for rank_idx in order[:top_k]:
                    freq = abs(angles[rank_idx]) / (2 * np.pi)
                    growth = float(np.real(np.log(eigvals[rank_idx] + 1e-12)))
                    dmd_modes.append(
                        {
                            "mode": int(rank_idx),
                            "frequency": float(freq),
                            "magnitude": float(mode_energy[rank_idx]),
                            "growth_rate": growth,
                        }
                    )
        except Exception as exc:  # pragma: no cover - defensive
            dmd_modes = [{"error": str(exc)}]

        rms_energy = float(np.sqrt(np.mean(np.square(energy_series)))) if energy_series.size else 0.0

        return {
            "available": True,
            "sample_count": sample_count,
            "window_size": window_size,
            "rms_energy": rms_energy,
            "fft": {
                "frequencies": freqs.tolist(),
                "avg_magnitude": positive_magnitude.tolist(),
                "peak_frequencies": peak_frequencies,
            },
            "dmd": {
                "modes": dmd_modes,
                "rank_used": int(min(len(energy_series) - 1, dmd_rank or len(energy_series))),
            },
            "energy_series": energy_series.tolist(),
        }
    def _quick_coherence_check(self, window: int = 32) -> float:
        """A fast check: Is the recent energy oscillating rhythmically?"""
        if len(self._delta_energy) < window:
            return 0.0
        
        # Get recent energy levels
        energy = np.array(list(self._delta_energy)[-window:])
        
        # Simple autocorrelation: Does the signal match itself shifted?
        # This detects a beat without needing a full Fourier Transform.
        centered = energy - np.mean(energy)
        if np.std(centered) < 1e-6:
            return 0.0 # Flatline
            
        autocorr = np.correlate(centered, centered, mode='full')
        # Normalize
        autocorr = autocorr[autocorr.size // 2:] / (np.var(centered) * len(centered))
        
        # Look for the second peak (the "echo" of the rhythm)
        # If the second peak is high, it means there is a strong beat.
        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i-1] < autocorr[i] > autocorr[i+1]:
                peaks.append(autocorr[i])
        
        if not peaks:
            return 0.0
            
        return float(max(peaks)) # The strength of the heartbeat

    # ------------------------------------------------------------------ public
    def is_ready(self) -> bool:
        return self._components is not None

    @property
    def basis_size(self) -> int:
        if self._components is None:
            return 0
        return self._components.shape[0]

    def observe(self, hidden_vector: Any, metadata: Optional[Dict[str, Any]] = None) -> Optional[DeltaObservation]:
        """Feed a new hidden state and update the Δ manifold.
        
        Args:
            hidden_vector: The hidden state vector to track.
            metadata: Optional interaction metadata (e.g., keystrokes, click actions).
        """
        current = self._normalize(hidden_vector)
        if current is None:
            return None
        observation = None
        if self._last_vector is not None:
            delta_raw = current - self._last_vector
            delta_norm = float(np.linalg.norm(delta_raw))
            if delta_norm > 1e-6:
                delta = delta_raw / delta_norm
                self._buffer.append(delta)
                self._delta_energy.append(delta_norm)
                self._needs_recompute = True
                self._observations_since_recompute += 1

                if self._should_recompute():
                    self._recompute_basis()

                observation = self._summarize(
                    delta,
                    delta_norm,
                    metadata,
                    coherence_fn=self._quick_coherence_check,
                )
                self._maybe_prune_basis()
                if observation is not None:
                    self._log_interaction(metadata, observation)

        self._last_vector = current
        return observation

    def save(self, filepath: str) -> None:
        """Persist the current basis (if any) to disk."""
        data: Dict[str, Any] = {
            "vector_dim": self.vector_dim,
            "window_size": self.window_size,
            "min_samples": self.min_samples,
            "variance_threshold": self.variance_threshold,
            "bias_scale": self.bias_scale,
            "components": self._components,
            "explained_variance": self._explained_variance,
            "total_variance": self._total_variance,
            "mean": self._mean,
        }
        np.save(filepath, data, allow_pickle=True)

    @classmethod
    def load(cls, filepath: str) -> "DeltaManifold":
        payload = np.load(filepath, allow_pickle=True).item()
        manifold = cls(
            vector_dim=int(payload["vector_dim"]),
            window_size=int(payload["window_size"]),
            min_samples=int(payload["min_samples"]),
            variance_threshold=float(payload["variance_threshold"]),
            bias_scale=float(payload["bias_scale"]),
        )
        manifold._components = payload.get("components")
        manifold._explained_variance = payload.get("explained_variance")
        manifold._total_variance = float(payload.get("total_variance", 0.0))
        manifold._mean = payload.get("mean")
        
        # Mark basis as inherited so warm start logic applies
        if manifold._components is not None:
            manifold._basis_is_inherited = True
            manifold._observations_since_recompute = 0
        
        return manifold

    # ----------------------------------------------------------------- helpers
    def _normalize(self, vector_input: Any) -> Optional[np.ndarray]:
        if vector_input is None:
            return None

        vector = np.asarray(vector_input, dtype=np.float32).reshape(-1)
        if vector.size != self.vector_dim:
            # Auto-adjust to new dimensionality (clears past basis).
            self._reset(vector.size)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return None
        return vector / norm

    def _reset(self, new_dim: int) -> None:
        self.vector_dim = new_dim
        self._buffer.clear()
        self._delta_energy.clear()
        self._components = None
        self._explained_variance = None
        self._total_variance = 0.0
        self._mean = None
        self._needs_recompute = False
        self._last_vector = None
        self._interaction_log = deque(maxlen=self.window_size)

    def _should_recompute(self) -> bool:
        if not self._needs_recompute:
            return False
        
        # Warm start: If we have an inherited basis, wait for substantial new data
        if self._basis_is_inherited:
            # Need at least 1/4 of window size before trusting new PCA over inherited basis
            if len(self._buffer) < self.window_size // 4:
                return False
            # Also require some observation time to see if inherited basis is working
            if self._observations_since_recompute < 8:
                return False
        
        return len(self._buffer) >= self.min_samples

    def _recompute_basis(self) -> None:
        matrix = np.stack(list(self._buffer), axis=0)
        mean = matrix.mean(axis=0, keepdims=True)
        centered = matrix - mean

        try:
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            self._needs_recompute = False
            return

        var = (s ** 2) / max(1, matrix.shape[0] - 1)
        total_var = float(var.sum())
        cumulative = np.cumsum(var) / (total_var + 1e-12)
        cutoff = np.searchsorted(cumulative, self.variance_threshold) + 1
        cutoff = int(min(cutoff, vt.shape[0]))

        new_components = vt[:cutoff]
        new_variance = var[:cutoff]
        
        if self._components is not None and self._components.size > 0:
            # We already have a basis (possibly inherited). Keep existing directions and
            # only append genuinely new ones so we do not collapse to a tiny PCA from
            # a short window.
            existing = self._components
            existing_var = (
                self._explained_variance
                if self._explained_variance is not None
                else np.ones(len(existing), dtype=np.float32)
            )

            new_directions = []
            new_variances = []

            for i, new_comp in enumerate(new_components):
                projections = existing @ new_comp
                captured_norm = np.linalg.norm(projections)

                # If the new component is mostly outside the current span, keep it.
                if captured_norm < 0.8:
                    orthogonal = new_comp - existing.T @ projections
                    norm = np.linalg.norm(orthogonal)
                    if norm > 1e-6:
                        new_directions.append(orthogonal / norm)
                        new_variances.append(new_variance[i] * (1 - captured_norm**2))

            if new_directions:
                new_variances_arr = np.asarray(new_variances, dtype=existing_var.dtype)
                old_size = len(existing)
                self._components = np.vstack([existing, new_directions])
                self._explained_variance = np.concatenate([existing_var, new_variances_arr])

                # Extend usage tracking for new dimensions
                if self._basis_usage is not None:
                    new_usage = np.zeros(len(new_directions), dtype=self._basis_usage.dtype)
                    self._basis_usage = np.concatenate([self._basis_usage, new_usage])

                print(f"Δ basis: retained {old_size}, added {len(new_directions)} new directions")
            else:
                # Keep the existing basis; nothing novel found this round
                self._components = existing
                self._explained_variance = existing_var
                print(f"Δ basis: retained {len(existing)} directions; no new additions")
        else:
            # No existing basis; just use the fresh PCA result
            self._components = new_components
            self._explained_variance = new_variance
            self._basis_usage = None
        
        self._total_variance = total_var
        self._mean = mean.squeeze(axis=0)
        self._needs_recompute = False
        self._basis_is_inherited = False  # No longer purely inherited after recompute
        self._observations_since_recompute = 0  # Reset counter

    def _summarize(
        self,
        delta: np.ndarray,
        delta_norm: float,
        metadata: Optional[Dict[str, Any]],
        coherence_fn: Optional[Callable[[], float]] = None,
    ) -> Optional[DeltaObservation]:
        if self._components is None or self._components.size == 0:
            return DeltaObservation(
                delta_norm=delta_norm,
                residual=1.0,
                projections=np.zeros(0, dtype=np.float32),
                basis_size=0,
                bias_vector=None,
                coherence=coherence_fn() if coherence_fn is not None else 0.0,
                metadata=metadata,
            )

        centered = delta - (self._mean if self._mean is not None else 0.0)
        projections = self._components @ centered
        
        # Track usage: decay old usage, add current projection magnitudes
        if self._basis_usage is None:
            self._basis_usage = np.abs(projections)
        else:
            # Safety check: if basis size changed, reinitialize usage tracking
            if len(self._basis_usage) != len(projections):
                print(f"Δ warning: basis_usage size mismatch ({len(self._basis_usage)} vs {len(projections)}), reinitializing")
                self._basis_usage = np.abs(projections)
            else:
                self._basis_usage = self._usage_decay * self._basis_usage + np.abs(projections)
        
        reconstructed = self._components.T @ projections
        residual_vec = centered - reconstructed
        residual = float(np.linalg.norm(residual_vec))

        bias_vector = self._build_bias_vector(projections)
        coherence_score = 0.0
        if coherence_fn is not None and residual > 0.5:
            coherence_score = coherence_fn()

        return DeltaObservation(
            delta_norm=delta_norm,
            residual=residual,
            projections=projections,
            basis_size=self._components.shape[0],
            bias_vector=bias_vector,
            coherence=coherence_score,
            metadata=metadata,
        )

    def _build_bias_vector(self, projections: np.ndarray) -> Optional[np.ndarray]:
        if self._components is None or self._components.size == 0:
            return None

        scaled = projections * self.bias_scale
        bias = self._components.T @ scaled

        norm = float(np.linalg.norm(bias))
        if norm <= 1e-8:
            return None
        return bias / norm

    def prune_unused_dimensions(self, usage_threshold: float = 0.05) -> int:
        """Prune basis dimensions that have very low usage over time.
        
        Returns number of dimensions pruned.
        """
        if self._components is None or self._basis_usage is None:
            return 0
        
        if len(self._basis_usage) == 0:
            return 0
        
        # Normalize usage to [0, 1]
        usage_normalized = self._basis_usage / (self._basis_usage.max() + 1e-12)
        
        # Keep dimensions above threshold
        keep_mask = usage_normalized >= usage_threshold
        num_kept = np.sum(keep_mask)
        num_pruned = len(self._components) - num_kept
        
        if num_pruned > 0 and num_kept > 0:
            self._components = self._components[keep_mask]
            self._explained_variance = self._explained_variance[keep_mask] if self._explained_variance is not None else None
            self._basis_usage = self._basis_usage[keep_mask]
            print(f"Δ basis: pruned {num_pruned} unused dimensions, kept {num_kept}")
        
        return num_pruned

    def _maybe_prune_basis(self) -> None:
        """Periodic pruning to prevent unbounded basis growth."""
        if (
            self.prune_interval <= 0
            or self._components is None
            or self._basis_usage is None
            or len(self._components) == 0
        ):
            return

        self._since_last_prune += 1
        if self._since_last_prune < self.prune_interval:
            return

        pruned = self.prune_unused_dimensions(self.prune_usage_threshold)
        self._since_last_prune = 0
        if pruned > 0:
            print(f"Δ basis: auto-pruned {pruned} dims after {self.prune_interval} observations")

    # --------------------------------------------------------- metadata logging
    def _log_interaction(
        self,
        metadata: Optional[Dict[str, Any]],
        observation: DeltaObservation,
    ) -> None:
        """Record lightweight metadata + metric snapshot for auditability."""
        entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "delta_norm": float(observation.delta_norm),
            "residual": float(observation.residual),
            "basis_size": int(observation.basis_size),
            "projection_energy": float(np.linalg.norm(observation.projections))
            if observation.projections is not None and observation.projections.size > 0
            else 0.0,
            "bias_available": observation.bias_vector is not None,
            "keystrokes": None,
            "click_actions": None,
            "metadata": metadata or {},
        }

        if metadata:
            entry["keystrokes"] = metadata.get("keystrokes")
            entry["click_actions"] = (
                metadata.get("click_actions")
                or metadata.get("clicks")
                or metadata.get("implicit_clicks")
            )

        self._interaction_log.append(entry)
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as exc:
                # Logging failure should not disrupt runtime; degrade gracefully.
                print(f"Δ warning: could not append interaction log ({exc})")

    def get_interaction_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the recent interaction metadata log for downstream analysis."""
        if limit is not None and limit > 0:
            return list(self._interaction_log)[-limit:]
        return list(self._interaction_log)

"""
Geometric cross-session analyzer
Generates UMAPs colored by geometric class (Constriction, Cyclicality, etc.)
and compares structural labels to regex-based semantic cues.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
except Exception:  # Plotly is optional; handle at runtime
    go = None
    pyo = None

try:
    import umap
except Exception:
    umap = None

try:
    # SentenceTransformer is optional; used for zero-shot motif similarity
    from sentence_transformers import SentenceTransformer

    _DEFAULT_ZS_MODEL = "all-MiniLM-L6-v2"
except Exception:
    SentenceTransformer = None
    _DEFAULT_ZS_MODEL = None

from analysis.cross_session_analyzer import CrossSessionAnalyzer


@dataclass
class MarkerRecord:
    """Lightweight container for marker data used in geometric classification."""

    vector: np.ndarray
    text: str
    session_id: str
    exchange_num: int
    layer: str


GEOMETRIC_CLASSES: Dict[str, Dict[str, Any]] = {
    "Constriction": {
        "description": "Long, thin, non-branching lines",
        "regex": re.compile(
            r"\b(corridor|tunnel|tight|bottleneck|chute|canal|alley|gutter|hallway|hinge|crease)\b",
            re.IGNORECASE,
        ),
        "color": "#1f77b4",
    },
    "Cyclicality": {
        "description": "Circular or ring-like coils",
        "regex": re.compile(
            r"\b(spiral|circle|loop|orbit|spin|echo|round|whirlpool|coil|strip|filament)\b",
            re.IGNORECASE,
        ),
        "color": "#bcbd22",
    },
    "Divergence": {
        "description": "Diffuse clouds / expansion",
        "regex": re.compile(
            r"\b(ocean|void|vast|sky|infinite|expanse|space|abyss|field|horizon|canvas)\b",
            re.IGNORECASE,
        ),
        "color": "#e377c2",
    },
    "Fragmentation": {
        "description": "Multiple separated clusters",
        "regex": re.compile(
            r"\b(island|archipelago|shard|broken|fragment|piece|scattered|cluster|patchwork|pulse)\b",
            re.IGNORECASE,
        ),
        "color": "#ff7f0e",
    },
    "Convergence": {
        "description": "Star-shaped pull toward a center",
        "regex": re.compile(
            r"\b(gravity|pull|magnet|focus|anchor|funnel|attractor|sink|inner|seed|map)\b",
            re.IGNORECASE,
        ),
        "color": "#2ca02c",
    },
}


class GeometricPatternAnalyzer:
    """Classify cross-session embeddings by geometric motif and compare to semantics."""

    def __init__(
        self,
        cross_analyzer: CrossSessionAnalyzer,
        output_dir: Optional[Path] = None,
        zero_shot: bool = False,
        zero_shot_model: Optional[str] = None,
    ):
        self.cross_analyzer = cross_analyzer
        self.output_dir = Path(
            output_dir or cross_analyzer.sessions_path / "geometric_analysis"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zero_shot = zero_shot and SentenceTransformer is not None
        self.zero_shot_model_name = zero_shot_model or _DEFAULT_ZS_MODEL
        self._zs_model = None

    # === Public API ======================================================
    def run(self) -> Dict[str, Any]:
        """Run geometric analysis for each layer and write artifacts to disk."""
        results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
            "layers": {},
        }

        if umap is None:
            print("✗ UMAP not installed. Install with: pip install umap-learn")
            return results

        for layer in ["early", "mid", "late"]:
            print(f"\n=== Geometric analysis for {layer} layer ===")
            layer_result = self._analyze_layer(layer)
            if layer_result:
                results["layers"][layer] = layer_result

        if results["layers"]:
            with open(self.output_dir / "geometric_report.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n✓ Geometric report saved to {self.output_dir/'geometric_report.json'}")
        else:
            print("No layers produced results (insufficient markers?)")

        return results

    # === Core analysis ===================================================
    def _analyze_layer(self, layer: str) -> Optional[Dict[str, Any]]:
        markers = self._collect_markers(layer)
        if len(markers) < 10:
            print(f"  Skipping {layer}: need at least 10 markers, found {len(markers)}")
            return None

        vectors = np.stack([m.vector for m in markers])
        embedding_2d = self._compute_umap(vectors, n_components=2)
        embedding_3d = self._compute_umap(vectors, n_components=3)

        labels, eps = self._cluster_points(embedding_2d)
        cluster_ids = sorted(set(labels))
        cluster_count = len([c for c in cluster_ids if c != -1]) or 1

        print(f"  Markers: {len(markers)} | Clusters: {cluster_count} | eps={eps:.3f}")

        geom_labels = []
        semantic_labels_regex = []
        semantic_labels_zs = []
        cluster_summaries: List[Dict[str, Any]] = []

        for cid in cluster_ids:
            idxs = [i for i, lbl in enumerate(labels) if lbl == cid]
            if not idxs:
                continue
            cluster_points = embedding_2d[idxs]
            geom_label, metrics, score_vec = self._classify_geometry(
                cluster_points, cluster_count
            )
            for idx in idxs:
                geom_labels.append((idx, geom_label))
            sem_counts = Counter()
            sem_counts_zs = Counter()
            for idx in idxs:
                sem_label = self._semantic_label_regex(markers[idx].text)
                semantic_labels_regex.append((idx, sem_label))
                sem_counts[sem_label] += 1
                if self.zero_shot:
                    sem_zs = self._semantic_label_zs(markers[idx].text)
                    semantic_labels_zs.append((idx, sem_zs))
                    sem_counts_zs[sem_zs] += 1

            cluster_summaries.append(
                {
                    "cluster_id": int(cid),
                    "size": len(idxs),
                    "geometry_class": geom_label,
                    "semantic_tally_regex": dict(sem_counts),
                    "semantic_tally_zs": dict(sem_counts_zs) if self.zero_shot else {},
                    "metrics": metrics,
                    "geometry_scores": score_vec,
                    "sample_text": markers[idxs[0]].text[:200],
                }
            )

        # Sort labels back into index order
        geom_labels_sorted = [gl[1] for gl in sorted(geom_labels, key=lambda x: x[0])]
        semantic_sorted_regex = [
            sl[1] for sl in sorted(semantic_labels_regex, key=lambda x: x[0])
        ]
        semantic_sorted_zs: List[str] = []
        if self.zero_shot:
            semantic_sorted_zs = [
                sl[1] for sl in sorted(semantic_labels_zs, key=lambda x: x[0])
            ]

        alignment_regex = defaultdict(Counter)
        for g, s in zip(geom_labels_sorted, semantic_sorted_regex):
            alignment_regex[g][s] += 1

        alignment_stats_regex = self._alignment_statistics(
            geom_labels_sorted, semantic_sorted_regex
        )

        alignment_stats_zs = {}
        alignment_zs = {}
        if self.zero_shot and semantic_sorted_zs:
            alignment_zs = defaultdict(Counter)
            for g, s in zip(geom_labels_sorted, semantic_sorted_zs):
                alignment_zs[g][s] += 1
            alignment_stats_zs = self._alignment_statistics(
                geom_labels_sorted, semantic_sorted_zs
            )

        layer_dir = self.output_dir / layer
        layer_dir.mkdir(parents=True, exist_ok=True)

        paths = self._make_plots(
            layer_dir,
            layer,
            embedding_2d,
            embedding_3d,
            markers,
            geom_labels_sorted,
            semantic_sorted_regex,
            semantic_sorted_zs if semantic_sorted_zs else None,
        )

        return {
            "marker_count": len(markers),
            "cluster_count": cluster_count,
            "clusters": cluster_summaries,
            "alignment_regex": {
                g: dict(counts) for g, counts in alignment_regex.items()
            },
            "alignment_stats_regex": alignment_stats_regex,
            "alignment_zs": {g: dict(counts) for g, counts in alignment_zs.items()}
            if alignment_zs
            else {},
            "alignment_stats_zs": alignment_stats_zs,
            "files": paths,
        }

    # === Helpers =========================================================
    def _collect_markers(self, layer: str) -> List[MarkerRecord]:
        """Gather markers for a specific layer from the cross-session analyzer."""
        markers: List[MarkerRecord] = []
        combine_text = getattr(self.cross_analyzer, "_combine_marker_text", None)

        for session in self.cross_analyzer.loaded_sessions:
            session_id = session.get("session_info", {}).get("session_id", "unknown")
            for marker in session.get("identity_markers", []):
                try:
                    if marker.get("layer") != layer:
                        continue
                    vector = marker.get("vector")
                    if vector is None:
                        continue
                    vec = np.array(vector)
                    if vec.size == 0:
                        continue
                    text = (
                        combine_text(marker)
                        if callable(combine_text)
                        else marker.get("text", "")
                    ) or ""
                    exchange = marker.get("exchange_num", 0)
                    markers.append(
                        MarkerRecord(
                            vector=vec.astype(float),
                            text=text,
                            session_id=session_id,
                            exchange_num=exchange,
                            layer=layer,
                        )
                    )
                except Exception:
                    continue
        return markers

    def _compute_umap(self, vectors: np.ndarray, n_components: int) -> np.ndarray:
        reducer = umap.UMAP(
            n_neighbors=min(15, len(vectors) - 1),
            min_dist=0.1,
            n_components=n_components,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(vectors)

    def _cluster_points(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Cluster points with a distance-adaptive DBSCAN."""
        if len(points) < 5:
            return np.zeros(len(points), dtype=int), 0.0

        n_neighbors = min(8, len(points) - 1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(points)
        distances, _ = nbrs.kneighbors(points)
        kth_dist = distances[:, -1]
        eps = max(np.percentile(kth_dist, 80), 0.05)

        db = DBSCAN(eps=eps, min_samples=max(3, n_neighbors))
        labels = db.fit_predict(points)
        return labels, eps

    def _classify_geometry(
        self, points: np.ndarray, cluster_count: int
    ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """Assign a geometric motif using simple shape statistics."""
        centroid = np.mean(points, axis=0)
        radii = np.linalg.norm(points - centroid, axis=1)
        radius_mean = float(np.mean(radii)) + 1e-8
        radius_std = float(np.std(radii))
        radius_cv = radius_std / radius_mean
        center_frac = float(np.mean(radii <= np.percentile(radii, 30)))
        outer_frac = float(np.mean(radii >= np.percentile(radii, 70)))

        pca = PCA(n_components=min(2, points.shape[1]))
        pca.fit(points)
        var_ratio = pca.explained_variance_ratio_
        width_ratio = float(var_ratio[1] / (var_ratio[0] + 1e-8)) if len(var_ratio) > 1 else 0.0

        # Geometry cues with explicit gates to avoid Divergence dominance.
        linearity_score = float(np.clip(1.0 - width_ratio, 0.0, 1.0))

        # Curvature via angle changes along an MST path to catch coils.
        curvature = self._avg_turning_angle(points)
        ring_score = float(np.clip(curvature / math.pi, 0.0, 1.0))

        spread = float(np.linalg.norm(points.ptp(axis=0)))
        divergence_core = float(np.clip(np.tanh(spread) * (0.5 + 0.5 * min(1.5, radius_cv)), 0.0, 1.0))
        divergence_score = divergence_core if (spread > 0.8 and radius_cv > 0.15) else 0.2 * divergence_core

        # Treat fragmentation as a weak tie-breaker based on global cluster count,
        # not a dominant per-cluster metric.
        if cluster_count <= 2:
            fragmentation_score = 0.0
        else:
            fragmentation_score = float(np.clip((cluster_count - 2) / 10.0, 0.0, 0.4))

        # Dense hub: many points near center -> convergence.
        convergence_score = float(np.clip(center_frac * 1.2 + outer_frac * 0.3, 0.0, 1.0))

        scores = {
            "Constriction": linearity_score,
            "Cyclicality": ring_score,
            "Divergence": divergence_score,
            "Fragmentation": fragmentation_score,
            "Convergence": convergence_score,
        }
        top_label = max(scores.items(), key=lambda x: x[1])[0]

        metrics = {
            "radius_mean": radius_mean,
            "radius_std": radius_std,
            "radius_cv": radius_cv,
            "center_fraction": center_frac,
            "outer_fraction": outer_frac,
            "width_ratio": width_ratio,
            "curvature": curvature,
            "linearity_score": linearity_score,
            "ring_score": ring_score,
            "divergence_score": divergence_score,
            "fragmentation_score": fragmentation_score,
            "convergence_score": convergence_score,
        }
        return top_label, metrics, scores

    def _semantic_label_regex(self, text: str) -> str:
        for name, cfg in GEOMETRIC_CLASSES.items():
            if cfg["regex"].search(text or ""):
                return name
        return "No-match"

    def _semantic_label_zs(self, text: str) -> str:
        model = self._get_zs_model()
        if not model:
            return "No-match"
        # Encode text and class descriptions; pick closest
        class_texts = [v["description"] for v in GEOMETRIC_CLASSES.values()]
        labels = list(GEOMETRIC_CLASSES.keys())
        try:
            text_emb = model.encode([text], normalize_embeddings=True)
            class_emb = model.encode(class_texts, normalize_embeddings=True)
            sims = np.dot(text_emb, class_emb.T)[0]
            best_idx = int(np.argmax(sims))
            return labels[best_idx]
        except Exception:
            return "No-match"

    def _get_zs_model(self):
        if not self.zero_shot or not SentenceTransformer:
            return None
        if self._zs_model is None:
            try:
                self._zs_model = SentenceTransformer(self.zero_shot_model_name)
            except Exception as exc:
                print(f"  Warning: zero-shot model load failed: {exc}")
                self._zs_model = None
        return self._zs_model

    def _alignment_statistics(
        self, geom_labels: List[str], semantic_labels: List[str]
    ) -> Dict[str, Any]:
        """Quantify correlation between geometry and regex semantics."""
        assert len(geom_labels) == len(semantic_labels)

        n = len(geom_labels)
        agreement = sum(
            1 for g, s in zip(geom_labels, semantic_labels) if s != "No-match" and g == s
        )
        known_sem = sum(1 for s in semantic_labels if s != "No-match")
        agreement_rate = agreement / known_sem if known_sem else 0.0

        mi = mutual_info_score(geom_labels, semantic_labels) if n else 0.0
        nmi = normalized_mutual_info_score(geom_labels, semantic_labels) if n else 0.0

        geom_to_sem = defaultdict(Counter)
        for g, s in zip(geom_labels, semantic_labels):
            geom_to_sem[g][s] += 1

        dominant = {}
        for g, counts in geom_to_sem.items():
            total = sum(counts.values())
            top = counts.most_common(1)[0] if total else ("", 0)
            dominant[g] = {
                "top_semantic": top[0],
                "top_fraction": top[1] / total if total else 0.0,
                "total": total,
            }

        return {
            "samples": n,
            "known_semantic_samples": known_sem,
            "agreement_rate": agreement_rate,
            "mutual_information": mi,
            "normalized_mutual_information": nmi,
            "dominant_mapping": dominant,
        }

    def _avg_turning_angle(self, points: np.ndarray) -> float:
        """Estimate curvature via angle changes along an MST path."""
        if len(points) < 3:
            return 0.0
        try:
            from scipy.spatial import distance_matrix
        except Exception:
            return 0.0

        dist = distance_matrix(points, points)
        n = len(points)
        visited = [False] * n
        visited[0] = True
        edges = []

        for _ in range(n - 1):
            min_edge = None
            for i in range(n):
                if not visited[i]:
                    continue
                for j in range(n):
                    if visited[j]:
                        continue
                    if min_edge is None or dist[i, j] < min_edge[0]:
                        min_edge = (dist[i, j], i, j)
            if min_edge is None:
                break
            _, i, j = min_edge
            visited[j] = True
            edges.append((i, j))

        # Build a path greedily from MST edges
        graph = defaultdict(list)
        for i, j in edges:
            graph[i].append(j)
            graph[j].append(i)
        start = max(graph.keys(), key=lambda k: len(graph[k]))
        path = [start]
        seen = {start}
        while True:
            neighbors = [n for n in graph[path[-1]] if n not in seen]
            if not neighbors:
                break
            nxt = neighbors[0]
            path.append(nxt)
            seen.add(nxt)

        if len(path) < 3:
            return 0.0

        angles = []
        pts = points[path]
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[i - 1]
            v2 = pts[i + 1] - pts[i]
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
            cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
            angles.append(math.acos(cos_angle))

        return float(np.mean(angles)) if angles else 0.0

    # === Visualization ===================================================
    def _make_plots(
        self,
        layer_dir: Path,
        layer: str,
        embedding_2d: np.ndarray,
        embedding_3d: np.ndarray,
        markers: List[MarkerRecord],
        geom_labels: List[str],
        semantic_labels: List[str],
        semantic_labels_zs: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        files: Dict[str, str] = {}

        # Static 2D plot: geometry vs semantics
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        self._scatter2d(
            axes[0],
            embedding_2d,
            geom_labels,
            title=f"{layer.upper()} — Geometry classes",
            palette={k: v["color"] for k, v in GEOMETRIC_CLASSES.items()},
        )
        semantic_palette = self._build_semantic_palette(semantic_labels)
        self._scatter2d(
            axes[1],
            embedding_2d,
            semantic_labels,
            title=f"{layer.upper()} — Regex semantic hits",
            palette=semantic_palette,
        )
        plt.tight_layout()
        png_path = layer_dir / "umap_geometry.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        files["umap_2d"] = str(png_path)

        # Static 3D plot
        fig3d = plt.figure(figsize=(10, 8))
        ax3d = fig3d.add_subplot(111, projection="3d")
        palette = {k: v["color"] for k, v in GEOMETRIC_CLASSES.items()}
        for label in set(geom_labels):
            mask = [g == label for g in geom_labels]
            color = palette.get(label, "#888888")
            ax3d.scatter(
                embedding_3d[mask, 0],
                embedding_3d[mask, 1],
                embedding_3d[mask, 2],
                s=25,
                alpha=0.75,
                c=color,
                label=label,
                depthshade=True,
            )
        ax3d.set_title(f"{layer.upper()} — Geometry (3D UMAP)")
        ax3d.set_xlabel("UMAP 1")
        ax3d.set_ylabel("UMAP 2")
        ax3d.set_zlabel("UMAP 3")
        ax3d.view_init(elev=22, azim=35)
        ax3d.legend()
        ax3d.grid(True, alpha=0.3)
        png3d_path = layer_dir / "umap_geometry_3d.png"
        fig3d.savefig(png3d_path, dpi=300, bbox_inches="tight")
        plt.close(fig3d)
        files["umap_3d"] = str(png3d_path)

        # Static 3D plot for regex semantics
        fig3d_sem = plt.figure(figsize=(10, 8))
        ax3d_sem = fig3d_sem.add_subplot(111, projection="3d")
        sem_palette = self._build_semantic_palette(semantic_labels)
        for label in set(semantic_labels):
            mask = [s == label for s in semantic_labels]
            ax3d_sem.scatter(
                embedding_3d[mask, 0],
                embedding_3d[mask, 1],
                embedding_3d[mask, 2],
                s=25,
                alpha=0.75,
                c=sem_palette.get(label, "#888888"),
                label=f"{label} ({sum(mask)})",
                depthshade=True,
            )
        ax3d_sem.set_title(f"{layer.upper()} — Regex semantics (3D UMAP)")
        ax3d_sem.set_xlabel("UMAP 1")
        ax3d_sem.set_ylabel("UMAP 2")
        ax3d_sem.set_zlabel("UMAP 3")
        ax3d_sem.view_init(elev=22, azim=35)
        ax3d_sem.legend(fontsize=8)
        ax3d_sem.grid(True, alpha=0.3)
        png3d_sem_path = layer_dir / "umap_semantics_regex_3d.png"
        fig3d_sem.savefig(png3d_sem_path, dpi=300, bbox_inches="tight")
        plt.close(fig3d_sem)
        files["umap_3d_semantics_regex"] = str(png3d_sem_path)

        # Static 3D plot for zero-shot semantics if available
        if semantic_labels_zs:
            fig3d_zs = plt.figure(figsize=(10, 8))
            ax3d_zs = fig3d_zs.add_subplot(111, projection="3d")
            sem_palette_zs = self._build_semantic_palette(semantic_labels_zs)
            for label in set(semantic_labels_zs):
                mask = [s == label for s in semantic_labels_zs]
                ax3d_zs.scatter(
                    embedding_3d[mask, 0],
                    embedding_3d[mask, 1],
                    embedding_3d[mask, 2],
                    s=25,
                    alpha=0.75,
                    c=sem_palette_zs.get(label, "#888888"),
                    label=f"{label} ({sum(mask)})",
                    depthshade=True,
                )
            ax3d_zs.set_title(f"{layer.upper()} — Zero-shot semantics (3D UMAP)")
            ax3d_zs.set_xlabel("UMAP 1")
            ax3d_zs.set_ylabel("UMAP 2")
            ax3d_zs.set_zlabel("UMAP 3")
            ax3d_zs.view_init(elev=22, azim=35)
            ax3d_zs.legend(fontsize=8)
            ax3d_zs.grid(True, alpha=0.3)
            png3d_zs_path = layer_dir / "umap_semantics_zs_3d.png"
            fig3d_zs.savefig(png3d_zs_path, dpi=300, bbox_inches="tight")
            plt.close(fig3d_zs)
            files["umap_3d_semantics_zs"] = str(png3d_zs_path)

        # Interactive HTML (3D) if plotly is available
        if go and pyo:
            fig = go.Figure()
            palette = {k: v["color"] for k, v in GEOMETRIC_CLASSES.items()}
            for label in set(geom_labels):
                mask = [g == label for g in geom_labels]
                hover = [
                    f"{label} | {markers[i].session_id} #{markers[i].exchange_num}<br>"
                    f"Semantic: {semantic_labels[i]}"
                    for i, m in enumerate(mask) if m
                ]
                fig.add_trace(
                    go.Scatter3d(
                        x=embedding_3d[mask, 0],
                        y=embedding_3d[mask, 1],
                        z=embedding_3d[mask, 2],
                        mode="markers",
                        marker=dict(size=4, color=palette.get(label, "#888888"), opacity=0.85),
                        name=label,
                        hoverinfo="text",
                        hovertext=hover,
                    )
                )
            fig.update_layout(
                title=f"{layer.upper()} geometry vs semantics",
                scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
                margin=dict(l=10, r=10, b=10, t=40),
            )
            html_path = layer_dir / "umap_geometry_3d.html"
            pyo.plot(fig, filename=str(html_path), auto_open=False, include_plotlyjs="cdn")
            files["umap_3d_html"] = str(html_path)

            # Regex semantic interactive 3D
            fig_sem = go.Figure()
            sem_palette = self._build_semantic_palette(semantic_labels)
            for label in set(semantic_labels):
                mask = [s == label for s in semantic_labels]
                hover = [
                    f"{label} | {markers[i].session_id} #{markers[i].exchange_num}<br>"
                    f"Geom: {geom_labels[i]}"
                    for i, m in enumerate(mask) if m
                ]
                fig_sem.add_trace(
                    go.Scatter3d(
                        x=embedding_3d[mask, 0],
                        y=embedding_3d[mask, 1],
                        z=embedding_3d[mask, 2],
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=sem_palette.get(label, "#888888"),
                            opacity=0.85,
                        ),
                        name=label,
                        hoverinfo="text",
                        hovertext=hover,
                    )
                )
            fig_sem.update_layout(
                title=f"{layer.upper()} regex semantics (3D UMAP)",
                scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
                margin=dict(l=10, r=10, b=10, t=40),
            )
            html_sem_path = layer_dir / "umap_semantics_regex_3d.html"
            pyo.plot(fig_sem, filename=str(html_sem_path), auto_open=False, include_plotlyjs="cdn")
            files["umap_3d_html_semantics_regex"] = str(html_sem_path)

            # Zero-shot semantic interactive 3D
            if semantic_labels_zs:
                fig_sem_zs = go.Figure()
                sem_palette_zs = self._build_semantic_palette(semantic_labels_zs)
                for label in set(semantic_labels_zs):
                    mask = [s == label for s in semantic_labels_zs]
                    hover = [
                        f"{label} | {markers[i].session_id} #{markers[i].exchange_num}<br>"
                        f"Geom: {geom_labels[i]}"
                        for i, m in enumerate(mask) if m
                    ]
                    fig_sem_zs.add_trace(
                        go.Scatter3d(
                            x=embedding_3d[mask, 0],
                            y=embedding_3d[mask, 1],
                            z=embedding_3d[mask, 2],
                            mode="markers",
                            marker=dict(
                                size=4,
                                color=sem_palette_zs.get(label, "#888888"),
                                opacity=0.85,
                            ),
                                name=label,
                                hoverinfo="text",
                                hovertext=hover,
                        )
                    )
                fig_sem_zs.update_layout(
                    title=f"{layer.upper()} zero-shot semantics (3D UMAP)",
                    scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
                    margin=dict(l=10, r=10, b=10, t=40),
                )
                html_sem_zs_path = layer_dir / "umap_semantics_zs_3d.html"
                pyo.plot(fig_sem_zs, filename=str(html_sem_zs_path), auto_open=False, include_plotlyjs="cdn")
                files["umap_3d_html_semantics_zs"] = str(html_sem_zs_path)
        else:
            print("  (plotly not available; skipping interactive HTML)")

        return files

    def _build_semantic_palette(self, labels: Iterable[str]) -> Dict[str, str]:
        unique = sorted(set(labels))
        cmap = cm.get_cmap("tab20", len(unique))
        return {label: cm.colors.to_hex(cmap(i)) for i, label in enumerate(unique)}

    def _scatter2d(
        self,
        ax: plt.Axes,
        points: np.ndarray,
        labels: List[str],
        title: str,
        palette: Dict[str, str],
    ) -> None:
        for label in sorted(set(labels)):
            mask = [l == label for l in labels]
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=40,
                alpha=0.75,
                c=palette.get(label, "#888888"),
                label=f"{label} ({sum(mask)})",
            )
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")


__all__ = ["GeometricPatternAnalyzer", "GEOMETRIC_CLASSES"]

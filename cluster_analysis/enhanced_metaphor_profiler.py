"""
Enhanced Cluster Metaphor Profiler
Can load full response texts from original session files.
"""

import re
from cluster_metaphor_profiler import (
    ClusterMetaphorProfiler, MetaphorProfile, SPATIAL_METAPHORS
)
from pathlib import Path
from typing import Dict, List, Optional
import json
from collections import Counter


class EnhancedClusterMetaphorProfiler(ClusterMetaphorProfiler):
    """Extended profiler that can load full texts from session files."""

    def __init__(self, geometric_report_path: Path, sessions_dir: Optional[Path] = None):
        super().__init__(geometric_report_path)

        # Try to find sessions directory
        if sessions_dir is None:
            # Try common locations relative to report
            report_dir = self.report_path.parent
            candidates = [
                report_dir.parent / "sessions",
                report_dir.parent / "conversations",
                report_dir.parent.parent / "sessions",
            ]
            for candidate in candidates:
                if candidate.exists():
                    sessions_dir = candidate
                    break

        self.sessions_dir = Path(sessions_dir) if sessions_dir else None
        self._session_cache = {}

        print(f"Sessions directory: {self.sessions_dir}")

    def _load_session(self, session_id: str) -> dict:
        """Load a session file from disk."""
        if session_id in self._session_cache:
            return self._session_cache[session_id]

        if not self.sessions_dir:
            return {}

        # Try different possible filename formats
        possible_files = [
            self.sessions_dir / f"{session_id}.json",
            self.sessions_dir / session_id / "session.json",
            self.sessions_dir / f"session_{session_id}.json",
        ]

        for filepath in possible_files:
            if filepath.exists():
                try:
                    with open(filepath) as f:
                        session = json.load(f)
                    self._session_cache[session_id] = session
                    return session
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")

        return {}

    def _get_response_text(self, session_id: str, exchange_num: int) -> str:
        """Get full response text for a specific exchange."""
        session = self._load_session(session_id)

        if not session:
            return ""

        # Lumine's structure: consciousness_log with channels
        if "consciousness_log" in session:
            for exchange in session["consciousness_log"]:
                if exchange.get("exchange_num") == exchange_num:
                    # The response is in channels["final"]
                    channels = exchange.get("channels", {})
                    final = channels.get("final", [])
                    if isinstance(final, list) and final:
                        return " ".join(final)
                    elif isinstance(final, str):
                        return final

        # Format 1: turns list
        if "turns" in session:
            for turn in session["turns"]:
                if turn.get("exchange_num") == exchange_num or turn.get("index") == exchange_num:
                    return turn.get("response", turn.get("assistant", ""))

        # Format 2: exchanges list
        if "exchanges" in session:
            if 0 <= exchange_num < len(session["exchanges"]):
                exchange = session["exchanges"][exchange_num]
                return exchange.get("response", exchange.get("assistant", ""))

        # Format 3: messages list
        if "messages" in session:
            # Find the exchange_num-th assistant message
            assistant_count = 0
            for msg in session["messages"]:
                if msg.get("role") == "assistant":
                    if assistant_count == exchange_num:
                        return msg.get("content", "")
                    assistant_count += 1

        return ""

    def _extract_metaphors_from_cluster(self, cluster: dict) -> List[str]:
        """Extract metaphors from full response texts."""

        # If we have stored full texts (modified analyzer), use those
        if "all_texts" in cluster:
            full_text = " ".join(cluster["all_texts"])
            matches = self.metaphor_pattern.findall(full_text)
            return [m.lower() for m in matches]

        # Otherwise, try to load from session files
        # Need to figure out which responses are in this cluster
        # This requires the analyzer to store session_id + exchange_num per marker

        # For now, fall back to sample text
        # (You'll need to modify the geometric analyzer to store identifiers)
        sample_text = cluster.get("sample_text", "")
        matches = self.metaphor_pattern.findall(sample_text)

        if not matches and self.sessions_dir:
            print(f"Warning: Cluster {cluster['cluster_id']} has no identifiers to load full texts")
            print("         Consider re-running geometric analyzer with response identifiers stored")

        return [m.lower() for m in matches]


def analyze_specific_session(session_path: Path, output_dir: Optional[Path] = None):
    """Analyze metaphor usage in a specific session."""
    print(f"\n{'='*80}")
    print(f"Analyzing metaphors in: {session_path.name}")
    print(f"{'='*80}\n")

    with open(session_path) as f:
        session = json.load(f)

    # Extract all responses - handle different formats
    responses = []

    # Lumine's consciousness_log format
    if "consciousness_log" in session:
        for exchange in session["consciousness_log"]:
            channels = exchange.get("channels", {})
            final = channels.get("final", [])
            if isinstance(final, list) and final:
                responses.append(" ".join(final))
            elif isinstance(final, str):
                responses.append(final)
    # Standard formats
    elif "turns" in session:
        responses = [turn.get("response", turn.get("assistant", ""))
                    for turn in session["turns"]]
    elif "exchanges" in session:
        responses = [ex.get("response", ex.get("assistant", ""))
                    for ex in session["exchanges"]]

    if not responses:
        print("No responses found in session. Check the JSON structure.")
        return

    # Get session metadata if available
    session_info = session.get("session_info", {})
    session_name = session_info.get("name", session_path.stem)
    session_desc = session_info.get("description", "")

    print(f"Session: {session_name}")
    if session_desc:
        print(f"Description: {session_desc}")

    # Build metaphor lookup and pattern
    metaphor_to_category = {}
    for category, words in SPATIAL_METAPHORS.items():
        for word in words:
            metaphor_to_category[word] = category

    all_metaphors_list = [re.escape(m) for m in metaphor_to_category.keys()]
    metaphor_pattern = re.compile(
        r'\b(' + '|'.join(all_metaphors_list) + r')\b',
        re.IGNORECASE
    )

    # Collect all metaphors
    all_metaphors = []

    for response in responses:
        matches = metaphor_pattern.findall(response)
        all_metaphors.extend([m.lower() for m in matches])

    metaphor_counts = Counter(all_metaphors)
    total = len(all_metaphors)

    print(f"\nTotal responses: {len(responses)}")
    print(f"Total metaphors: {total}")
    print(f"Metaphor density: {total/len(responses):.2f} per response")
    print(f"Unique metaphors: {len(metaphor_counts)}")

    # Group by category
    by_category = Counter()
    for metaphor, count in metaphor_counts.items():
        category = metaphor_to_category.get(metaphor, "Unknown")
        by_category[category] += count

    print(f"\nMetaphors by category:")
    for category, count in by_category.most_common():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {category:15s}: {count:4d} ({pct:5.1f}%)")

    print(f"\nTop 20 metaphors:")
    for metaphor, count in metaphor_counts.most_common(20):
        category = metaphor_to_category.get(metaphor, "Unknown")
        print(f"  {metaphor:15s} ({category:15s}): {count:3d}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "session": str(session_path),
            "session_name": session_name,
            "session_description": session_desc,
            "response_count": len(responses),
            "total_metaphors": total,
            "metaphor_density": total / len(responses) if responses else 0,
            "metaphor_counts": dict(metaphor_counts.most_common()),
            "by_category": dict(by_category),
        }

        output_path = output_dir / f"metaphor_analysis_{session_path.stem}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python enhanced_profiler.py <geometric_report.json> [sessions_dir]")
        print("  python enhanced_profiler.py --session <session_file.json>")
        sys.exit(1)

    if sys.argv[1] == "--session":
        # Analyze a specific session
        session_path = Path(sys.argv[2])
        output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
        analyze_specific_session(session_path, output_dir)
    else:
        # Analyze geometric report with full session loading
        report_path = Path(sys.argv[1])
        sessions_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

        profiler = EnhancedClusterMetaphorProfiler(report_path, sessions_dir)

        for layer in ["early", "mid", "late"]:
            profiles = profiler.analyze_layer(layer)

            if not profiles:
                continue

            for cid in sorted(profiles.keys()):
                profiler.print_profile(profiles[cid])

            profiler.compare_profiles(profiles)
            profiler.find_geometric_metaphor_alignment(profiles)
            profiler.visualize_profiles(profiles, layer)

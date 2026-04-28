"""Simple research-gap heuristics."""

from __future__ import annotations

from collections import Counter


def detect_gaps(cluster_labels: list[int]) -> list[str]:
    """Flag sparse clusters as under-explored themes."""

    if not cluster_labels:
        return ["No clusters were generated. Try a broader topic or fetch more papers."]

    gaps: list[str] = []
    counts = Counter(cluster_labels)

    for label, count in sorted(counts.items()):
        if count <= 1:
            gaps.append(f"Cluster {label} is under-explored ({count} sentence).")

    if not gaps:
        gaps.append("No sparse cluster was detected in this small sample.")

    return gaps

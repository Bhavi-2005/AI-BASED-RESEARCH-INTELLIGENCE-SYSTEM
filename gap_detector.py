"""Simple research-gap heuristics."""

from __future__ import annotations

from collections import Counter




def detect_gaps(
    cluster_labels: list[int], 
    contradiction_results: list[dict[str, any]] = None,
    sentence_records: list[dict[str, any]] = None
) -> list[dict[str, any]]:
    """Mathematically validate gaps and provide evidence traceback."""

    if not cluster_labels:
        return [{"message": "No clusters were generated.", "is_gap": False}]

    counts = Counter(cluster_labels)
    total_samples = len(cluster_labels)
    
    # Map contradictions to clusters
    contra_data = {}
    if contradiction_results:
        for res in contradiction_results:
            if res["label"].lower() == "contradiction":
                for cl in [res["left_cluster"], res["right_cluster"]]:
                    contra_data.setdefault(cl, []).append(res)

    gaps: list[dict[str, any]] = []
    
    for label, count in sorted(counts.items()):
        # Formula: confidence = ((1 / (cluster_density + 1)) * 0.4 + contradiction_score * 0.4 + novelty_score * 0.2)
        
        density_score = 1 / (count + 1)
        
        # Contradiction score (0 to 1)
        cl_contras = contra_data.get(label, [])
        contra_score = min(len(cl_contras) * 0.35, 1.0)
        
        # Novelty score (simulated/heuristic based on cluster weight)
        novelty_score = 0.8 if count == 1 else (0.5 if count <= 3 else 0.2)
        
        final_confidence = (density_score * 0.4) + (contra_score * 0.4) + (novelty_score * 0.2)
        
        is_gap = final_confidence > 0.45 or count <= 2
        
        if is_gap:
            # Evidence Traceback
            supporting_papers = sorted(list({
                r["title"] for i, r in enumerate(sentence_records) if cluster_labels[i] == label
            }))
            
            reasons = []
            if density_score > 0.3: reasons.append("Low statistical density")
            if contra_score > 0: reasons.append("High conflict in findings")
            if novelty_score > 0.5: reasons.append("High novelty/Singularity")

            gaps.append({
                "label": label,
                "is_gap": True,
                "confidence": round(final_confidence, 2),
                "confidence_reasoning": f"Based on {len(supporting_papers)} source papers + {len(cl_contras)} contradiction signals.",
                "novelty_level": "High" if novelty_score > 0.6 else "Medium",
                "evidence_count": len(supporting_papers),
                "supporting_papers": supporting_papers,
                "contradicting_claims": cl_contras,
                "reasons": reasons,
                "summary": f"Thematic Gap in Cluster {label} ({reasons[0] if reasons else 'Unstable findings'})"
            })

    if not gaps:
        gaps.append({"message": "The field appears well-researched in this sample.", "is_gap": False})

    return gaps

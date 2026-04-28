"""Simple idea generation for detected research gaps."""

from __future__ import annotations




def generate_ideas(
    gaps: list[dict[str, any]],
    clustered_sentences: dict[int, list[str]],
    methods: list[str] = None
) -> list[dict[str, any]]:
    """Create structured 'Strategy Tree' ideas with multi-level improvements."""

    strategies: list[dict[str, any]] = []
    
    modern_context = "Transformer-based"
    if methods and any(m in ["BERT", "GPT", "Transformer", "T5"] for m in methods):
        modern_context = "Advanced Generative/Hybrid"

    for gap in gaps:
        if not gap.get("is_gap"):
            continue
            
        label = gap["label"]
        sentences = clustered_sentences.get(label, [])
        context = sentences[0] if sentences else "this domain"

        # Strategy Tree Branch
        strategies.append({
            "target": f"Cluster {label}",
            "primary_idea": f"Adapt {modern_context} models for {gap['reasons'][0].lower()}",
            "alternatives": [
                f"Low-resource distillation of existing {modern_context} models.",
                "Cross-modal verification using contrastive learning."
            ],
            "improvements": [
                "Layer-wise relevance propagation for explainability.",
                "Quantization for edge-device deployment."
            ],
            "impact_tags": ["Academic", "Industry"] if gap["novelty_level"] == "High" else ["Social", "Industry"],
            "difficulty": "Moderate" if len(sentences) > 0 else "Advanced"
        })

    if not strategies:
        strategies.append({
            "target": "Global Domain",
            "primary_idea": "Meta-analysis of historical trends vs modern architectures",
            "alternatives": ["Comparative benchmarking", "Hybrid rule-based fallback systems"],
            "improvements": ["Automated hyperparameter tuning"],
            "impact_tags": ["Academic"],
            "difficulty": "Standard"
        })

    return strategies[:6]

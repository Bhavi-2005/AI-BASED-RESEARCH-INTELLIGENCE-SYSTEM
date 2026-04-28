"""Simple idea generation for detected research gaps."""

from __future__ import annotations


def generate_ideas(
    gaps: list[str],
    clustered_sentences: dict[int, list[str]],
    topic_hint: str = "",
) -> list[str]:
    """Create lightweight next-step ideas from sparse clusters."""

    ideas: list[str] = []
    topic_prefix = f"In {topic_hint}, " if topic_hint.strip() else ""

    for label, sentences in sorted(clustered_sentences.items()):
        if len(sentences) > 1:
            continue

        anchor = sentences[0]
        ideas.append(
            f"{topic_prefix}explore Cluster {label} with modern deep learning methods, "
            f"using this theme as a starting point: {anchor}"
        )

    if not ideas:
        for gap in gaps:
            if "under-explored" in gap.lower():
                ideas.append(f"{topic_prefix}explore: {gap} using modern deep learning methods.")

    if not ideas:
        ideas.append(
            f"{topic_prefix}compare the strongest themes with newer architectures "
            "to test whether an overlooked sub-problem opens a publishable direction."
        )

    return ideas[:5]

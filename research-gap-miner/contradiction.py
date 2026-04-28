"""Contradiction detection with an MNLI model."""

from __future__ import annotations

from functools import lru_cache

from transformers import pipeline


DEFAULT_NLI_MODEL = "roberta-large-mnli"


@lru_cache(maxsize=1)
def get_nli_pipeline(model_name: str = DEFAULT_NLI_MODEL):
    return pipeline("text-classification", model=model_name)


def check_contradiction(sent1: str, sent2: str, model_name: str = DEFAULT_NLI_MODEL):
    """Return NLI labels for two statements."""

    nli = get_nli_pipeline(model_name)
    return nli(f"{sent1} </s></s> {sent2}", top_k=None)

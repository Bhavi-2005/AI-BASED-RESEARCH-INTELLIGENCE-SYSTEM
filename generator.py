"""Grounded answer generation and summarization."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import pipeline

from vector_store import SearchResult


DEFAULT_GENERATION_MODEL = "google/flan-t5-base"
DEFAULT_SUMMARY_MODEL = "google/flan-t5-base"


@dataclass(frozen=True)
class GeneratedAnswer:
    """Final model output returned to the UI."""

    answer: str
    retrieved_context: str
    chunk_summaries: list[str]



def load_research_generator(model_name: str = DEFAULT_GENERATION_MODEL):
    """Robust pipeline loader with auto-fallback."""
    try:
        # Standard Seq2Seq task for T5
        return pipeline("text2text-generation", model=model_name)
    except Exception as e:
        # Fallback to general text generation if Seq2Seq detection fails
        try:
            return pipeline("text-generation", model="gpt2")
        except:
             return None

class AnswerGenerator:
    """Generates grounded answers and research strategies."""

    def __init__(
        self,
        model_name: str = DEFAULT_GENERATION_MODEL,
        max_input_chars: int = 6000,
        max_new_tokens: int = 250,
    ) -> None:
        self.model_name = model_name
        self.max_input_chars = max_input_chars
        self.max_new_tokens = max_new_tokens
        self.generator = load_research_generator(model_name)

    def generate(
        self,
        query: str,
        results: list[SearchResult],
        is_relevant: bool,
    ) -> GeneratedAnswer:
        context = self._format_context(results)

        if not is_relevant or not self.generator:
            return GeneratedAnswer(
                answer=(
                    "I could not synthesize a grounded answer. The retrieved information "
                    "does not contain direct evidence for this specific query."
                ),
                retrieved_context=context,
                chunk_summaries=[],
            )

        # Research-Grade Prompting
        prompt = (
            "You are a PhD-level research assistant specializing in literature synthesis.\n"
            "Analyze the retrieved context below to answer the user query.\n"
            "REQUIREMENTS: Use ONLY provided evidence. If ambiguous, state 'Evidence insufficient'.\n\n"
            f"TEXT EVIDENCE:\n{context[:self.max_input_chars]}\n\n"
            f"QUERY: {query}\n\n"
            "STRICT GROUNDED ANSWER:"
        )
        
        response = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )[0]["generated_text"]

        # Clean-up response (remove prompt if model is decoder-only)
        if "STRICT GROUNDED ANSWER:" in response:
            response = response.split("STRICT GROUNDED ANSWER:")[-1]

        return GeneratedAnswer(
            answer=response.strip(),
            retrieved_context=context,
            chunk_summaries=self.summarize_chunks(results),
        )

    def summarize_chunks(self, results: list[SearchResult]) -> list[str]:
        """Synthesize rapid summaries of evidence blocks."""

        summaries: list[str] = []
        if not self.generator: return []

        for result in results:
            text = result.chunk.text[:1000]
            prompt = f"Summarize this research passage in one concise technical sentence:\n\n{text}"
            
            gen_out = self.generator(prompt, max_new_tokens=50, do_sample=False)[0]["generated_text"]
            
            if "technical sentence:" in gen_out:
                gen_out = gen_out.split("technical sentence:")[-1]
            
            summaries.append(gen_out.strip())

        return summaries

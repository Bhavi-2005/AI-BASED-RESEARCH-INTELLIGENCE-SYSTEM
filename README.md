# Context-Grounded RAG Question Answering

This project is a complete beginner-friendly Retrieval-Augmented Generation
system. It accepts PDF and text documents, splits them into overlapping chunks,
embeds those chunks with Sentence-BERT, indexes them in FAISS, retrieves the
most relevant context for a user query, and generates an answer grounded in that
retrieved context.

## Architecture

```text
User Query -> Sentence-BERT Embedding -> FAISS Search -> Top-k Context -> T5 Generator -> Answer
```

## Files

- `data_loader.py`: loads PDF/TXT files and splits text into 500-word chunks with 50-word overlap.
- `embedding.py`: wraps Sentence-BERT and caches embeddings under `.rag_cache/embeddings`.
- `vector_store.py`: builds, searches, saves, and loads a FAISS cosine-similarity index.
- `retriever.py`: embeds the query and returns top-k chunks with a confidence score.
- `generator.py`: uses FLAN-T5 to answer using only retrieved context and summarize chunks.
- `app.py`: Streamlit interface for uploading documents and asking questions.

## Setup

```bash
py -3.11 -m venv --system-site-packages .venv311
.venv311\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the embedding and generation models, so it may take a
few minutes.

## Example Input

Upload a text file containing:

```text
RAG systems combine information retrieval with language generation. They first
retrieve relevant source passages, then pass those passages to a language model
so the generated answer can be grounded in external knowledge.
```

Ask:

```text
How does a RAG system generate grounded answers?
```

Example output:

```text
A RAG system retrieves relevant source passages first and then gives those
passages to a language model, allowing the answer to be based on the retrieved
external knowledge.
```

The UI also shows retrieved source chunks, similarity scores, summaries, and a
confidence score.

## Multilingual Support

The default embedding model is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which supports
English and Tamil semantic retrieval. Generation quality depends on the selected
text-generation model. For stronger Tamil answers, replace the generator with a
multilingual instruction model that your machine can run.

## Evaluation

The app reports a relevance-style confidence score from the average cosine
similarity of the top retrieved chunks. This is not a full human accuracy score,
but it is a useful operational signal for whether the retrieved context matches
the question.

## Notes

- Answers are instructed to use only retrieved context.
- Irrelevant questions are handled gracefully when top similarity is below the
  relevance threshold.
- Embeddings and FAISS indexes are cached under `.rag_cache` to avoid repeated
  computation for the same documents.

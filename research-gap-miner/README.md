# Research Gap Miner

Research Gap Miner is a Streamlit mini-system that searches arXiv for a topic
or accepts uploaded PDFs, extracts text, generates sentence embeddings,
clusters themes, flags under-explored clusters as possible research gaps,
compares theme statements for contradictions, and suggests next-step ideas.

## Project Structure

```text
research-gap-miner/
├── app.py
├── fetch_papers.py
├── pdf_processor.py
├── embeddings.py
├── clustering.py
├── contradiction.py
├── gap_detector.py
└── requirements.txt
```

## Setup

```bash
cd research-gap-miner
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the embedding model, so it can take a few minutes.

## What It Does

1. The user enters a topic such as `NLP healthcare`.
2. The app fetches relevant arXiv papers.
3. It downloads and extracts text from each PDF.
4. It splits the text into sentences and embeds them with Sentence-BERT.
5. It clusters sentence embeddings into themes.
6. It reports sparse clusters as possible research gaps.
7. It compares representative theme statements with an NLI model.
8. It suggests idea directions from the detected gaps.

## Current UI Sections

- Upload PDFs or search topic
- Themes
- Research Gaps
- Suggested Ideas
- Contradiction Detection

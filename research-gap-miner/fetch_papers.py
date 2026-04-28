"""Fetch paper metadata from the arXiv API."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import requests


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def fetch_arxiv_papers(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Search arXiv and return paper titles with direct PDF links."""

    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    safe_query = quote_plus(cleaned_query)
    url = (
        f"{ARXIV_API_URL}?search_query=all:{safe_query}"
        f"&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    papers: list[dict[str, str]] = []

    for entry in root.findall(f"{ATOM_NS}entry"):
        title_node = entry.find(f"{ATOM_NS}title")
        title = " ".join((title_node.text or "Untitled paper").split())
        pdf_link = ""

        for link in entry.findall(f"{ATOM_NS}link"):
            if link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib.get("href", "")
                break

        if pdf_link:
            papers.append({"title": title, "pdf": pdf_link})

    return papers

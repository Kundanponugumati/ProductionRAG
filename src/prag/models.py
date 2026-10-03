"""Data passed between ingestion stages."""

import uuid
from pathlib import Path

from pydantic import BaseModel

# Fixed namespace so the same chunk always gets the same UUID across runs.
CHUNK_NAMESPACE = uuid.UUID("6f1c7d2e-3b4a-4c5d-9e8f-0a1b2c3d4e5f")


class Document(BaseModel):
    doc_name: str  # e.g. "3M_2018_10K"
    path: Path
    company: str
    doc_type: str  # "10k", "10q", "8k", "Earnings"
    doc_period: int  # fiscal year


class Page(BaseModel):
    doc_name: str
    page_num: int  # 0-indexed, same as FinanceBench evidence_page_num
    text: str


class Chunk(BaseModel):
    doc_name: str
    company: str
    doc_type: str
    doc_period: int
    page_num: int
    chunk_idx: int  # position within the page
    text: str
    token_count: int

    @property
    def id(self) -> str:
        """Deterministic point ID: re-ingesting overwrites instead of duplicating."""
        key = f"{self.doc_name}:{self.page_num}:{self.chunk_idx}"
        return str(uuid.uuid5(CHUNK_NAMESPACE, key))


class SearchResult(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity, higher = more similar

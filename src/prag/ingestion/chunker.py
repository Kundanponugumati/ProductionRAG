"""Fixed-size token windows with overlap, computed within a single page.

Chunks never cross page boundaries, so each chunk maps to exactly one page —
that's what lets us compare retrieval results against FinanceBench evidence pages.
"""

from tokenizers import Tokenizer

from prag.models import Chunk, Document, Page


class TokenChunker:
    def __init__(self, model: str, chunk_size: int, chunk_overlap: int) -> None:
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("need 0 <= chunk_overlap < chunk_size")
        # Same tokenizer as the embedding model, so chunk_size is a true token budget.
        self.tokenizer = Tokenizer.from_pretrained(model)
        self.tokenizer.no_truncation()
        self.chunk_size = chunk_size
        self.step = chunk_size - chunk_overlap

    def chunk_page(self, page: Page, doc: Document) -> list[Chunk]:
        # Offsets map each token back to its (start, end) characters in page.text.
        offsets = self.tokenizer.encode(page.text, add_special_tokens=False).offsets
        chunks = []
        for chunk_idx, start in enumerate(range(0, len(offsets), self.step)):
            window = offsets[start : start + self.chunk_size]
            chunks.append(
                Chunk(
                    doc_name=doc.doc_name,
                    company=doc.company,
                    doc_type=doc.doc_type,
                    doc_period=doc.doc_period,
                    page_num=page.page_num,
                    chunk_idx=chunk_idx,
                    # Slice the original text instead of decoding tokens: BGE's tokenizer
                    # lowercases and splits numbers ("$5,363" → "$ 5, 363").
                    text=page.text[window[0][0] : window[-1][1]],
                    token_count=len(window),
                )
            )
            if start + self.chunk_size >= len(offsets):
                break  # this window reached the end of the page
        return chunks

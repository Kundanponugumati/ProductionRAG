"""PDF → one Page per PDF page (plain text, no layout or table handling yet)."""

import pymupdf

from prag.models import Document, Page


def parse_pdf(doc: Document) -> list[Page]:
    # Use doc.path (or whichever attribute holds the string path to the file)
    with pymupdf.open(doc.path) as pdf:
        return [
            Page(doc_name=doc.doc_name, page_num=i, text=page.get_text())
            for i, page in enumerate(pdf.pages())
        ]

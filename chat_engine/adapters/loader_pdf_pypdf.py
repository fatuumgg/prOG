from __future__ import annotations

from pathlib import Path
from typing import Sequence

from chat_engine.domain.rag_models import LoadedPage
from chat_engine.ports.loaders import DocumentLoader

class PdfLoaderPyPDF(DocumentLoader):
    def load(self, path: str) -> Sequence[LoadedPage]:
        try:
            from pypdf import PdfReader 
        except Exception as e:
            raise RuntimeError("Для PDF нужен пакет pypdf: pip install pypdf") from e

        p = Path(path)
        reader = PdfReader(str(p))
        pages: list[LoadedPage] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(LoadedPage(source=str(p), text=text, page=i))
        return pages

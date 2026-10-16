import os
import uuid
from typing import List, Dict, Any, Tuple
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .schema import DocumentChunk
from . import config


class DocumentParser:
    """
    Modern layout-aware document parser.
    Extracts text, preserves tables, headers, and page numbers across formats.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )

    def parse_and_chunk(self, file_path: str) -> Tuple[List[DocumentChunk], int, str]:
        """
        Parses a document file, extracts structured markdown, and chunks it.
        
        Returns:
            Tuple of (chunks, page_count, full_preview)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        if file_ext == ".pdf":
            pages_data = self._parse_pdf(file_path)
        elif file_ext in [".docx", ".doc"]:
            pages_data = self._parse_docx(file_path)
        elif file_ext in [".pptx", ".ppt"]:
            pages_data = self._parse_pptx(file_path)
        elif file_ext in [".xlsx", ".xls"]:
            pages_data = self._parse_excel(file_path)
        elif file_ext in [".txt", ".md", ".csv", ".json"]:
            pages_data = self._parse_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Chunk the parsed pages while preserving page numbers and metadata
        chunks: List[DocumentChunk] = []
        preview_text = ""

        for page_info in pages_data:
            page_text = page_info["text"]
            page_num = page_info.get("page", 1)

            if not page_text or not page_text.strip():
                continue

            if len(preview_text) < 1000:
                preview_text += page_text + "\n"

            # Split text within the page to keep citations strictly linked to pages
            sub_chunks = self.text_splitter.split_text(page_text)
            for idx, chunk_text in enumerate(sub_chunks):
                chunk_id = f"{filename}_p{page_num}_c{idx}_{uuid.uuid4().hex[:6]}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text.strip(),
                    metadata={
                        "source": filename,
                        "file_path": file_path,
                        "file_type": file_ext,
                        "page": page_num,
                        "chunk_index": idx
                    }
                ))

        total_pages = max([p.get("page", 1) for p in pages_data], default=1)
        return chunks, total_pages, preview_text[:600]

    def _parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts structured Markdown from PDF using PyMuPDF4LLM.
        Accurately preserves tables as markdown tables and layout hierarchy.
        """
        import pymupdf4llm
        pages_markdown = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        
        pages_data = []
        for item in pages_markdown:
            page_num = item.get("metadata", {}).get("page", 0) + 1  # 1-indexed
            text = item.get("text", "")
            pages_data.append({"page": page_num, "text": text})

        return pages_data

    def _parse_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses DOCX documents into clean markdown including tables."""
        from docx import Document as DocxDocument

        doc = DocxDocument(file_path)
        full_text = []

        for p in doc.paragraphs:
            if p.text.strip():
                full_text.append(p.text)

        # Parse tables into markdown tables
        for table in doc.tables:
            table_md = []
            for i, row in enumerate(table.rows):
                row_text = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                table_md.append("| " + " | ".join(row_text) + " |")
                if i == 0:
                    table_md.append("| " + " | ".join(["---"] * len(row_text)) + " |")
            if table_md:
                full_text.append("\n" + "\n".join(table_md) + "\n")

        return [{"page": 1, "text": "\n\n".join(full_text)}]

    def _parse_pptx(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses PowerPoint presentations slide-by-slide natively.
        Eliminates the need for external Pandoc binaries.
        """
        from pptx import Presentation

        prs = Presentation(file_path)
        pages_data = []

        for slide_idx, slide in enumerate(prs.slides, start=1):
            slide_content = []
            
            # Slide title and shapes
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            slide_content.append(text)
                
                # Slide tables
                elif shape.has_table:
                    table_md = []
                    table = shape.table
                    for r_idx, row in enumerate(table.rows):
                        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                        table_md.append("| " + " | ".join(cells) + " |")
                        if r_idx == 0:
                            table_md.append("| " + " | ".join(["---"] * len(cells)) + " |")
                    if table_md:
                        slide_content.append("\n" + "\n".join(table_md) + "\n")

            # Slide notes (if any)
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    slide_content.append(f"**Presenter Notes:** {notes}")

            full_slide_text = f"## Slide {slide_idx}\n" + "\n".join(slide_content)
            pages_data.append({"page": slide_idx, "text": full_slide_text})

        return pages_data

    def _parse_excel(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses Excel spreadsheets, converting each sheet into a Markdown table."""
        xl = pd.read_excel(file_path, sheet_name=None)
        pages_data = []

        for sheet_idx, (sheet_name, df) in enumerate(xl.items(), start=1):
            # Drop completely empty rows and columns
            clean_df = df.dropna(how="all").dropna(axis=1, how="all")
            # Convert to markdown table
            sheet_md = f"## Sheet: {sheet_name}\n\n"
            sheet_md += clean_df.to_markdown(index=False)
            pages_data.append({"page": sheet_idx, "text": sheet_md})

        return pages_data

    def _parse_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses plain text / markdown files."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return [{"page": 1, "text": content}]

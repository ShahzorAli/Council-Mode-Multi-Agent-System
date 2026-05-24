"""
Council Mode - Document Loader
Parses PDF documents using PyMuPDF and extracts structured text.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF

from utils.logger import get_logger

logger = get_logger("retrieval")


@dataclass
class DocumentChunk:
    """A chunk of text from a parsed document."""
    
    text: str
    source_file: str
    page_number: int
    chunk_id: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def page_content(self) -> str:
        """Alias for text for compatibility with standard RAG tools."""
        return self.text
    
    def __str__(self):
        return f"[{self.source_file} | Page {self.page_number}] {self.text[:100]}..."


@dataclass
class ParsedDocument:
    """A fully parsed document with its pages."""
    
    filename: str
    filepath: str
    total_pages: int
    pages: List[dict] = field(default_factory=list)  # {"page_num": int, "text": str}
    
    @property
    def full_text(self) -> str:
        return "\n\n".join(p["text"] for p in self.pages)


class DocumentLoader:
    """
    Loads and parses PDF documents using PyMuPDF.
    
    Features:
    - Extracts text from all pages
    - Preserves page-level metadata for citation tracking
    - Handles multi-column layouts
    - Supports batch loading from a directory
    """
    
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".json"}
    
    def __init__(self):
        self.loaded_documents: List[ParsedDocument] = []
    
    def load_file(self, filepath: str | Path) -> ParsedDocument:
        """Load a single file based on its extension."""
        filepath = Path(filepath)
        ext = filepath.suffix.lower()
        
        if ext == ".pdf":
            return self.load_pdf(filepath)
        elif ext == ".txt":
            return self.load_txt(filepath)
        elif ext == ".json":
            return self.load_json(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def load_pdf(self, filepath: str | Path) -> ParsedDocument:
        """
        Parse a single PDF file.
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            ParsedDocument with extracted text and metadata
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Document not found: {filepath}")
        
        logger.info(f"Parsing PDF: {filepath.name}")
        
        doc = fitz.open(str(filepath))
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # Clean up extracted text
            text = self._clean_text(text)
            
            if text.strip():  # Only add non-empty pages
                pages.append({
                    "page_num": page_num + 1,  # 1-indexed
                    "text": text,
                })
        
        doc.close()
        
        parsed = ParsedDocument(
            filename=filepath.name,
            filepath=str(filepath),
            total_pages=len(pages),
            pages=pages,
        )
        
        self.loaded_documents.append(parsed)
        logger.info(
            f"Parsed PDF '{filepath.name}': {parsed.total_pages} pages"
        )
        
        return parsed

    def load_txt(self, filepath: str | Path) -> ParsedDocument:
        """Load a text file."""
        filepath = Path(filepath)
        logger.info(f"Loading TXT: {filepath.name}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        text = self._clean_text(text)
        
        parsed = ParsedDocument(
            filename=filepath.name,
            filepath=str(filepath),
            total_pages=1,
            pages=[{"page_num": 1, "text": text}],
        )
        
        self.loaded_documents.append(parsed)
        return parsed

    def load_json(self, filepath: str | Path) -> ParsedDocument:
        """
        Load a JSON or JSONL file. 
        Attempts to find 'text' or 'context' fields, otherwise serializes whole object.
        """
        import json
        filepath = Path(filepath)
        logger.info(f"Loading JSON: {filepath.name}")
        
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                # Try loading as a single JSON object/list
                data = json.load(f)
            except json.JSONDecodeError:
                # Fallback: Try loading as JSONL (one JSON object per line)
                f.seek(0)
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
        # Heuristic: extract text from the parsed data
        text_parts = []
        
        def extract_text(obj):
            if isinstance(obj, dict):
                # Priority keys
                for key in ["text", "context", "body", "content", "question"]:
                    if key in obj and isinstance(obj[key], str):
                        text_parts.append(obj[key])
                        # If it has both question and answer, join them
                        if "answer" in obj and isinstance(obj["answer"], str):
                            text_parts.append(obj["answer"])
                        return
                # Fallback: join all string values
                for v in obj.values():
                    if isinstance(v, str) and len(v) > 20:
                        text_parts.append(v)
                    elif isinstance(v, (dict, list)):
                        extract_text(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text(item)
        
        extract_text(data)
        
        full_text = "\n\n".join(text_parts) if text_parts else json.dumps(data, indent=2)
        full_text = self._clean_text(full_text)
        
        parsed = ParsedDocument(
            filename=filepath.name,
            filepath=str(filepath),
            total_pages=1,
            pages=[{"page_num": 1, "text": full_text}],
        )
        
        self.loaded_documents.append(parsed)
        return parsed

    def load_directory(self, directory: str | Path) -> List[ParsedDocument]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of ParsedDocument objects
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents = []
        
        # Look for all supported extensions
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(list(directory.glob(f"*{ext}")))
            
        for filepath in sorted(files):
            try:
                doc = self.load_file(filepath)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load {filepath.name}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text — normalize whitespace, remove artifacts."""
        import re
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'\x00', '', text)  # Null bytes
        text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f]', '', text)  # Control chars
        
        # Restore paragraph breaks (heuristic: period followed by capital letter)
        text = re.sub(r'\. ([A-Z])', r'.\n\n\1', text)
        
        return text.strip()

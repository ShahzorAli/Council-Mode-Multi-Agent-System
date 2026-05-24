"""
Council Mode - Text Splitter
Splits documents into semantically meaningful chunks for embedding.
"""

from typing import List
from dataclasses import dataclass

from rag.document_loader import ParsedDocument, DocumentChunk
from utils.logger import get_logger

logger = get_logger("retrieval")


class TextSplitter:
    """
    Splits documents into overlapping chunks suitable for embedding and retrieval.
    
    Uses a recursive character-based splitting strategy that tries to split
    on natural boundaries (paragraphs → sentences → words).
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Target number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Separators in order of preference (try paragraph first, then sentence, etc.)
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    
    def split_document(self, document: ParsedDocument) -> List[DocumentChunk]:
        """
        Split a parsed document into chunks, preserving page-level provenance.
        
        Args:
            document: A ParsedDocument from the DocumentLoader
            
        Returns:
            List of DocumentChunk objects with source tracking
        """
        all_chunks = []
        
        for page in document.pages:
            page_text = page["text"]
            page_num = page["page_num"]
            
            # Split this page's text into chunks
            text_chunks = self._recursive_split(page_text)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    text=chunk_text,
                    source_file=document.filename,
                    page_number=page_num,
                    chunk_id=f"{document.filename}::p{page_num}::c{i}",
                    metadata={
                        "filepath": document.filepath,
                        "total_pages": document.total_pages,
                        "chunk_index": i,
                    }
                )
                all_chunks.append(chunk)
        
        logger.info(
            f"Split '{document.filename}' into {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        
        return all_chunks
    
    def split_documents(self, documents: List[ParsedDocument]) -> List[DocumentChunk]:
        """Split multiple documents into chunks."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        
        logger.info(f"Total chunks from {len(documents)} documents: {len(all_chunks)}")
        return all_chunks
    
    def _recursive_split(self, text: str, separators: List[str] = None) -> List[str]:
        """
        Recursively split text using a hierarchy of separators.
        Tries paragraph splits first, then sentence, then word boundaries.
        """
        if separators is None:
            separators = self.separators

        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        # Try each separator in order
        for i, separator in enumerate(separators):
            if separator in text:
                return self._split_with_separator(text, separator, separators[i+1:])
        
        # Fallback: hard split at chunk_size
        return self._hard_split(text)
    
    def _split_with_separator(self, text: str, separator: str, next_separators: List[str]) -> List[str]:
        """Split text on a separator, then merge small chunks."""
        parts = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            candidate = current_chunk + separator + part if current_chunk else part
            
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + separator + part
                else:
                    current_chunk = part
                
                # If single part exceeds chunk_size, recurse with finer separator
                if len(current_chunk) > self.chunk_size:
                    sub_chunks = self._recursive_split(current_chunk, next_separators)
                    if sub_chunks:
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1] if sub_chunks else ""
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _hard_split(self, text: str) -> List[str]:
        """Last resort: split at exact character boundaries."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks

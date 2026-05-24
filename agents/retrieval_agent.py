from typing import List, Optional, Tuple

from rag.retriever import Retriever
from rag.document_loader import DocumentChunk
from utils.logger import get_logger

logger = get_logger("retrieval")


class RetrievalAgent:
    """
    Agentic-RAG & Evidence Ingestion Agent 
    
    Unlike standard RAG (single-pass retrieval), this agent supports:
    - Initial retrieval for the first debate round
    - Iterative re-retrieval when the Skeptic identifies knowledge gaps
    - Refined search queries based on debate feedback
    - Deduplication across multiple retrieval passes
    """
    
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.all_evidence = ""
        self.all_chunks: List[DocumentChunk] = []
        self.retrieval_rounds = 0
    
    def initial_retrieve(self, query: str) -> Tuple[str, List[DocumentChunk]]:
        """Perform initial evidence retrieval for a query."""
        logger.info(f"Initial retrieval for: '{query[:80]}...'")
        self.retrieval_rounds = 1
        self.all_evidence, self.all_chunks = self.retriever.retrieve(query)
        return self.all_evidence, self.all_chunks
    
    def re_retrieve(self, refined_queries: List[str]) -> Tuple[str, List[DocumentChunk]]:
        """
        Perform iterative re-retrieval with refined search queries.
        Called when the Skeptic identifies knowledge gaps during debate.
        """
        self.retrieval_rounds += 1
        logger.info(
            f"Iterative re-retrieval (round {self.retrieval_rounds}) "
            f"with {len(refined_queries)} refined queries"
        )
        
        new_evidence, new_chunks = self.retriever.iterative_retrieve(refined_queries)
        
        # Merge with existing evidence (dedup by chunk_id)
        existing_ids = {c.chunk_id for c in self.all_chunks}
        added = 0
        for chunk in new_chunks:
            if chunk.chunk_id not in existing_ids:
                self.all_chunks.append(chunk)
                existing_ids.add(chunk.chunk_id)
                added += 1
        
        # Rebuild formatted evidence
        self.all_evidence = self._format_all_evidence()
        
        logger.info(f"Re-retrieval added {added} new chunks (total: {len(self.all_chunks)})")
        return self.all_evidence, self.all_chunks
    
    def _format_all_evidence(self) -> str:
        parts = []
        for i, chunk in enumerate(self.all_chunks, 1):
            parts.append(
                f"[Source {i}]\n"
                f"  File: {chunk.source_file} | Page: {chunk.page_number}\n"
                f"  Content: {chunk.text}\n"
            )
        return "\n".join(parts)
    
    def get_source_summary(self) -> str:
        """Get summary of all sources used."""
        sources = {}
        for c in self.all_chunks:
            key = c.source_file
            if key not in sources:
                sources[key] = set()
            sources[key].add(c.page_number)
        
        lines = [f"Sources used ({len(self.all_chunks)} chunks from {len(sources)} files):"]
        for f, pages in sources.items():
            lines.append(f"  - {f}: pages {sorted(pages)}")
        return "\n".join(lines)
    
    def reset(self):
        self.all_evidence = ""
        self.all_chunks = []
        self.retrieval_rounds = 0

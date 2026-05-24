"""
Council Mode - Retriever
High-level retrieval interface that combines vector search with formatting.
"""

from typing import List, Tuple, Optional

from rag.vector_store import VectorStore
from rag.document_loader import DocumentChunk
from utils.logger import get_logger

logger = get_logger("retrieval")


class Retriever:
    """
    High-level retrieval interface for the Council Mode system.

    """
    
    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k
        self.retrieval_history: List[dict] = []  # Track all queries for the session
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Tuple[str, List[DocumentChunk]]:
        """
        Retrieve relevant evidence for a query.
        """
        k = top_k or self.top_k
        results = self.vector_store.search(query, top_k=k)
        
        if not results:
            logger.warning(f"No evidence found for query: {query[:80]}...")
            return "No relevant evidence found in the knowledge base.", []
        
        # Format evidence for LLM consumption
        evidence_parts = []
        chunks = []
        
        for i, (chunk, score) in enumerate(results, 1):
            evidence_parts.append(
                f"[Source {i}] (Relevance: {score:.3f})\n"
                f"  File: {chunk.source_file} | Page: {chunk.page_number}\n"
                f"  Content: {chunk.text}\n"
            )
            chunks.append(chunk)
        
        formatted_evidence = "\n".join(evidence_parts)
        
        # Track retrieval history
        self.retrieval_history.append({
            "query": query,
            "num_results": len(results),
            "top_score": results[0][1] if results else 0,
            "sources": [c.source_file for c in chunks],
        })
        
        logger.info(
            f"Retrieved {len(results)} evidence chunks "
            f"(top score: {results[0][1]:.4f})"
        )
        
        return formatted_evidence, chunks
    
    def iterative_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> Tuple[str, List[DocumentChunk]]:
        """
        Perform iterative retrieval with multiple queries (for re-query during debate).
        
        Deduplicates results across queries.
        
        Args:
            queries: List of search queries (original + refined)
            top_k: Results per query
            
        Returns:
            Tuple of (combined_formatted_evidence, deduplicated_chunks)
        """
        all_chunks_map = {}  # chunk_id -> (chunk, best_score)
        
        for query in queries:
            results = self.vector_store.search(query, top_k=top_k or self.top_k)
            
            for chunk, score in results:
                if chunk.chunk_id not in all_chunks_map:
                    all_chunks_map[chunk.chunk_id] = (chunk, score)
                else:
                    # Keep the higher score
                    existing_score = all_chunks_map[chunk.chunk_id][1]
                    if score > existing_score:
                        all_chunks_map[chunk.chunk_id] = (chunk, score)
        
        # Sort by score descending
        sorted_results = sorted(
            all_chunks_map.values(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        # Format evidence
        evidence_parts = []
        chunks = []
        
        for i, (chunk, score) in enumerate(sorted_results, 1):
            evidence_parts.append(
                f"[Source {i}] (Relevance: {score:.3f})\n"
                f"  File: {chunk.source_file} | Page: {chunk.page_number}\n"
                f"  Content: {chunk.text}\n"
            )
            chunks.append(chunk)
        
        formatted_evidence = "\n".join(evidence_parts)
        
        logger.info(
            f"Iterative retrieval with {len(queries)} queries: "
            f"{len(sorted_results)} unique chunks"
        )
        
        return formatted_evidence, chunks
    
    def get_retrieval_summary(self) -> str:
        """Get a summary of all retrievals performed in this session."""
        if not self.retrieval_history:
            return "No retrievals performed."
        
        lines = [f"Total retrievals: {len(self.retrieval_history)}"]
        for i, entry in enumerate(self.retrieval_history, 1):
            lines.append(
                f"  {i}. Query: '{entry['query'][:50]}...' → "
                f"{entry['num_results']} results "
                f"(top: {entry['top_score']:.3f})"
            )
        
        return "\n".join(lines)

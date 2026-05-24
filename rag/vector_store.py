"""
Council Mode - Vector Store
FAISS-based vector database for document embeddings.
"""

import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from rag.document_loader import DocumentChunk
from utils.logger import get_logger

logger = get_logger("retrieval")


class VectorStore:
    """
    FAISS-based vector store for document chunk embeddings.
    
    Features:
    - Uses sentence-transformers for embedding generation
    - FAISS IndexFlatIP (Inner Product / Cosine Similarity) for search
    - Persists index + metadata to disk
    - Supports incremental document addition
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: Optional[str | Path] = None,
    ):
        """
        Args:
            embedding_model: HuggingFace sentence-transformers model name
            persist_dir: Directory to save/load the FAISS index
        """
        self.embedding_model_name = embedding_model
        self.persist_dir = Path(persist_dir) if persist_dir else None
        
        logger.info(f"Loading embedding model: {embedding_model}")
        
        # Hardware acceleration detection
        import torch
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
            
        logger.info(f"Using device: {device}")
        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index (Inner Product for cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Metadata storage: maps index position → DocumentChunk info
        self.chunks: List[DocumentChunk] = []
        
        # Try to load existing index
        if self.persist_dir and self._index_exists():
            self._load()
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """
        Embed and add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects to embed and store
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        # Extract texts for embedding
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,  # Normalize for cosine similarity via IP
            batch_size=128,             # Increase batch size for performance
        )
        
        # Add to FAISS index
        embeddings_np = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_np)
        
        # Store chunk metadata
        self.chunks.extend(chunks)
        
        logger.info(
            f"Added {len(chunks)} chunks to vector store "
            f"(total: {self.index.ntotal} vectors)"
        )
        
        # Persist to disk
        if self.persist_dir:
            self._save()
    
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for relevant chunks by semantic similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of (DocumentChunk, similarity_score) tuples, sorted by relevance
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty — no documents to search")
            return []
        
        # Embed the query
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
        )
        query_np = np.array(query_embedding, dtype=np.float32)
        
        # Search FAISS index
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_np, k)
        
        # Map results back to chunks
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and idx >= 0:
                results.append((self.chunks[idx], float(score)))
        
        logger.info(
            f"Search for '{query[:50]}...' returned {len(results)} results "
            f"(top score: {results[0][1]:.4f})" if results else 
            f"Search for '{query[:50]}...' returned 0 results"
        )
        
        return results
    
    def clear(self):
        """Clear all vectors and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        logger.info("Vector store cleared")
    
    @property
    def total_vectors(self) -> int:
        return self.index.ntotal
    
    @property
    def all_chunks(self) -> List[DocumentChunk]:
        """Return all chunks stored in the vector store."""
        return self.chunks
    
    def _save(self):
        """Persist FAISS index and metadata to disk."""
        if not self.persist_dir:
            return
        
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = self.persist_dir / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save chunk metadata (pickle for simplicity)
        meta_path = self.persist_dir / "chunks.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"Vector store saved to {self.persist_dir}")
    
    def _load(self):
        """Load FAISS index and metadata from disk."""
        index_path = self.persist_dir / "index.faiss"
        meta_path = self.persist_dir / "chunks.pkl"
        
        if index_path.exists() and meta_path.exists():
            self.index = faiss.read_index(str(index_path))
            
            with open(meta_path, "rb") as f:
                self.chunks = pickle.load(f)
            
            logger.info(
                f"Loaded vector store: {self.index.ntotal} vectors, "
                f"{len(self.chunks)} chunks"
            )
    
    def _index_exists(self) -> bool:
        """Check if a persisted index exists."""
        if not self.persist_dir:
            return False
        return (self.persist_dir / "index.faiss").exists()

"""
Council Mode - Configuration Management
Loads settings from .env and provides typed configuration objects.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()


class ModelConfig(BaseModel):
    """Configuration for LLM model assignments."""
    
    # Triage (small, fast model)

    triage_model: str = Field(
        default=os.getenv("TRIAGE_MODEL", "groq:llama-3.1-8b-instant"),
        description="Model for query triage (Fast/Low Cost)"
    )
    
    # Experts (Distributed Load)
    expert_model_1: str = Field(
        default=os.getenv("EXPERT_MODEL_1", "groq:llama-3.1-8b-instant"),
        description="First expert (Groq)"
    )
    expert_model_2: str = Field(
        default=os.getenv("EXPERT_MODEL_2", "groq:llama-3.1-8b-instant"),
        description="Second expert (Groq)"
    )
    expert_model_3: str = Field(
        default=os.getenv("EXPERT_MODEL_3", "gemini:gemini-2.0-flash"),
        description="Third expert (Gemini)"
    )
    
    # Heavy Lifting (Balanced Load)
    skeptic_model: str = Field(
        default=os.getenv("SKEPTIC_MODEL", "gemini:gemini-2.0-flash"),
        description="Skeptic (Gemini)"
    )
    synthesizer_model: str = Field(
        default=os.getenv("SYNTHESIZER_MODEL", "groq:llama-3.3-70b-versatile"),
        description="Final Synthesizer (Groq 70B)"
    )


class RAGConfig(BaseModel):
    """Configuration for RAG pipeline."""
    
    embedding_model: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        description="Sentence transformer model for embeddings"
    )
    chunk_size: int = Field(
        default=int(os.getenv("CHUNK_SIZE", "512")),
        description="Text chunk size for splitting"
    )
    chunk_overlap: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", "50")),
        description="Overlap between text chunks"
    )
    top_k: int = Field(
        default=int(os.getenv("TOP_K_RETRIEVAL", "5")),
        description="Number of top results to retrieve"
    )
    documents_dir: Path = Field(
        default=PROJECT_ROOT / os.getenv("DOCUMENTS_DIR", "data/documents"),
        description="Directory for source documents"
    )
    vector_db_dir: Path = Field(
        default=PROJECT_ROOT / os.getenv("VECTOR_DB_DIR", "data/vector_db"),
        description="Directory for FAISS index"
    )

class DebateConfig(BaseModel):
    """Configuration for the Multi-Agent Debate protocol."""
    
    num_experts: int = Field(
        default=int(os.getenv("NUM_EXPERTS", "3")),
        description="Number of expert agents to participate in the debate (1-3)"
    )
    max_rounds: int = Field(
        default=int(os.getenv("MAX_DEBATE_ROUNDS", "3")),
        description="Maximum number of debate rounds"
    )
    consensus_threshold: float = Field(
        default=float(os.getenv("CONSENSUS_THRESHOLD", "0.7")),
        description="Minimum agreement ratio to reach consensus (0.0 - 1.0)"
    )


class OllamaConfig(BaseModel):
    """Configuration for Ollama local models."""
    
    base_url: str = Field(
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        description="Ollama server base URL"
    )


class GroqConfig(BaseModel):
    """Configuration for Groq cloud inference API."""
    
    api_key: str = Field(
        default=os.getenv("GROQ_API_KEY", ""),
        description="Groq API key"
    )

class OpenRouterConfig(BaseModel):
    """Configuration for OpenRouter API."""
    api_key: str = Field(
        default=os.getenv("OPENROUTER_API_KEY", ""),
        description="OpenRouter API key"
    )

class DeepSeekConfig(BaseModel):
    """Configuration for DeepSeek API."""
    api_key: str = Field(
        default=os.getenv("DEEPSEEK_API_KEY", ""),
        description="DeepSeek API key"
    )

class CerebrasConfig(BaseModel):
    """Configuration for Cerebras API."""
    api_key: str = Field(
        default=os.getenv("CEREBRAS_API_KEY", ""),
        description="Cerebras API key"
    )

class GeminiConfig(BaseModel):
    """Configuration for Google Gemini API."""
    
    api_key: str = Field(
        default=os.getenv("GEMINI_API_KEY", ""),
        description="Google Gemini API key"
    )

class NvidiaConfig(BaseModel):
    """Configuration for NVIDIA NIM API."""
    
    api_key: str = Field(
        default=os.getenv("NVIDIA_API_KEY", ""),
        description="NVIDIA API key"
    )


# ============================================
# Global Configuration Instances
# ============================================
model_config = ModelConfig()
rag_config = RAGConfig()
debate_config = DebateConfig()
ollama_config = OllamaConfig()
gemini_config = GeminiConfig()
groq_config = GroqConfig()
openrouter_config = OpenRouterConfig()
deepseek_config = DeepSeekConfig()
cerebras_config = CerebrasConfig()
nvidia_config = NvidiaConfig()


def validate_config():
    """Validate that essential configuration is present."""
    warnings = []
    
    # Check Groq API key if any model uses Groq
    all_models = [
        model_config.triage_model,
        model_config.expert_model_1,
        model_config.expert_model_2,
        model_config.expert_model_3,
        model_config.skeptic_model,
        model_config.synthesizer_model,
    ]
    if any("groq" in m for m in all_models) and not groq_config.api_key:
        warnings.append("WARNING: GROQ_API_KEY not set -- Groq-based agents will fail.")
    
    # Check Gemini API key if any model uses Gemini
    if any("gemini" in m for m in all_models) and not gemini_config.api_key:
        warnings.append("WARNING: GEMINI_API_KEY not set -- Gemini-based agents will fail.")

    if any("openrouter" in m for m in all_models) and not openrouter_config.api_key:
        warnings.append("WARNING: OPENROUTER_API_KEY not set.")
    
    if any("deepseek" in m for m in all_models) and not deepseek_config.api_key:
        warnings.append("WARNING: DEEPSEEK_API_KEY not set.")

    if any("cerebras" in m for m in all_models) and not cerebras_config.api_key:
        warnings.append("WARNING: CEREBRAS_API_KEY not set.")
    
    if any("nvidia" in m for m in all_models) and not nvidia_config.api_key:
        warnings.append("WARNING: NVIDIA_API_KEY not set.")
    
    # Check directories exist
    if not rag_config.documents_dir.exists():
        rag_config.documents_dir.mkdir(parents=True, exist_ok=True)
        warnings.append(f"Created documents directory: {rag_config.documents_dir}")
    
    if not rag_config.vector_db_dir.exists():
        rag_config.vector_db_dir.mkdir(parents=True, exist_ok=True)
        warnings.append(f"Created vector DB directory: {rag_config.vector_db_dir}")
    
    return warnings

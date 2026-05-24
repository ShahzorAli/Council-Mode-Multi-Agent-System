"""
Council Mode - Base Model Interface
Abstract base class that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelResponse:
    """Standardized response from any LLM provider."""
    
    text: str
    model_name: str
    provider: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    reasoning: Optional[str] = None
    raw_response: Optional[dict] = field(default=None, repr=False)
    
    def __str__(self):
        return self.text


class BaseModel(ABC):
    """Abstract base class for LLM model providers."""
    
    def __init__(self, model_name: str, provider: str):
        self.model_name = model_name
        self.provider = provider
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            prompt: The user prompt/query
            system_prompt: System-level instructions
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens in response
            
        Returns:
            Standardized ModelResponse object
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the model is currently available/reachable."""
        pass
    
    def __repr__(self):
        return f"{self.provider}:{self.model_name}"

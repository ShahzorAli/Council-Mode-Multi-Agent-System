"""
Council Mode - Ollama Model Provider
Interfaces with locally running Ollama models (Llama 3, Mistral, etc.)
"""

import time
import asyncio
from typing import Optional

import ollama as ollama_client

from models.base_model import BaseModel, ModelResponse
from utils.logger import get_logger

logger = get_logger("system")


class OllamaModel(BaseModel):
    """
    LLM provider for locally running Ollama models.
    
    Supports any model available through Ollama:
    - llama3.1:8b, llama3.1:70b
    - mistral:7b, mixtral:8x7b
    - gemma2:9b, etc.
    """
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        super().__init__(model_name=model_name, provider="ollama")
        self.base_url = base_url
        self.client = ollama_client.AsyncClient(host=base_url)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ModelResponse:
        """Generate response using Ollama's local model."""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        
        try:
            # Add a timeout to prevent long hangs if Ollama is unreachable
            response = await asyncio.wait_for(
                self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                ),
                timeout=60.0 # 1 minute max for local triage
            )
            
            latency = (time.time() - start_time) * 1000  # ms
            
            text = response["message"]["content"]
            tokens = response.get("eval_count", None)
            
            logger.info(
                f"Ollama [{self.model_name}] responded "
                f"({latency:.0f}ms, ~{tokens or '?'} tokens)"
            )
            
            return ModelResponse(
                text=text,
                model_name=self.model_name,
                provider="ollama",
                tokens_used=tokens,
                latency_ms=latency,
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Ollama [{self.model_name}] error: {e}")
            raise ConnectionError(
                f"Failed to get response from Ollama model '{self.model_name}': {e}"
            )
    
    async def is_available(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            models_response = await self.client.list()
            available_models = [
                m.get("name", m.get("model", "")) 
                for m in models_response.get("models", [])
            ]
            # Check if our model is in the list (partial match for tags)
            return any(self.model_name in m for m in available_models)
        except Exception:
            return False

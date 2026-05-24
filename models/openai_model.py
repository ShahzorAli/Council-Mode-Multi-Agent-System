"""
Council Mode - OpenAI-Compatible Model Provider
Supports OpenRouter, DeepSeek, Cerebras, and other OpenAI-compatible APIs.
"""

import time
import asyncio
from typing import Optional
from openai import OpenAI
import os

from models.base_model import BaseModel, ModelResponse
from utils.logger import get_logger

logger = get_logger("models")

class OpenAICompatibleModel(BaseModel):
    """Provider for OpenAI-compatible APIs."""
    
    def __init__(self, model_name: str, provider: str, base_url: str, api_key: str):
        super().__init__(model_name, provider)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ModelResponse:
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Run in executor since openai client is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            
            latency = (time.time() - start_time) * 1000
            
            return ModelResponse(
                text=response.choices[0].message.content,
                model_name=self.model_name,
                provider=self.provider,
                tokens_used=response.usage.total_tokens if response.usage else None,
                latency_ms=latency,
                raw_response=response.model_dump()
            )
            
        except Exception as e:
            logger.error(f"Error from {self.provider} model '{self.model_name}': {e}")
            raise ConnectionError(f"Failed to get response from {self.provider} model '{self.model_name}': {e}")

    async def is_available(self) -> bool:
        # Simple check: can we list models or just return true if key exists
        return True

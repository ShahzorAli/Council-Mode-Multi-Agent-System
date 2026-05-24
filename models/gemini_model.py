"""
Council Mode - Google Gemini Model Provider
Interfaces with Google's Gemini API for cloud-based inference.
"""

import time
from typing import Optional

from google import genai
from google.genai import types

from models.base_model import BaseModel, ModelResponse
from utils.logger import get_logger

logger = get_logger("system")


class GeminiModel(BaseModel):
    """
    LLM provider for Google Gemini API.
    
    Supports models like:
    - gemini-2.0-flash
    - gemini-1.5-pro
    - gemini-1.5-flash
    """
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name=model_name, provider="gemini")
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ModelResponse:
        """Generate response using Gemini API."""
        
        start_time = time.time()
        
        max_retries = 3
        base_delay = 10  # Start with 10 seconds delay
        
        for attempt in range(max_retries):
            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                
                if system_prompt:
                    config.system_instruction = system_prompt
                
                # Use async client to avoid blocking event loop
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                
                latency = (time.time() - start_time) * 1000  # ms
                
                text = response.text or ""
                
                # Extract token count if available
                tokens = None
                if response.usage_metadata:
                    tokens = response.usage_metadata.total_token_count
                
                logger.info(
                    f"Gemini [{self.model_name}] responded "
                    f"({latency:.0f}ms, ~{tokens or '?'} tokens)"
                )
                
                return ModelResponse(
                    text=text,
                    model_name=self.model_name,
                    provider="gemini",
                    tokens_used=tokens,
                    latency_ms=latency,
                    raw_response={"text": text},
                )
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    # Extract wait time if possible, else use exponential backoff
                    wait_time = base_delay * (2 ** attempt)
                    if "retry in" in error_str.lower():
                        import re
                        match = re.search(r"retry in ([\d\.]+)s", error_str.lower())
                        if match:
                            wait_time = float(match.group(1)) + 1.0 # Add 1s buffer
                    
                    logger.warning(f"Gemini [{self.model_name}] rate limit hit. Retrying in {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
                    import asyncio
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Gemini [{self.model_name}] error: {e}")
                    raise ConnectionError(
                        f"Failed to get response from Gemini model '{self.model_name}': {e}"
                    )
    
    async def is_available(self) -> bool:
        """Check if Gemini API is reachable with the provided key."""
        try:
            # Quick test with minimal content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Say 'ok'",
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            return bool(response.text)
        except Exception:
            return False

"""
Council Mode - Groq Model Provider
Interfaces with Groq's fast inference API.
Supports Llama 3, Mixtral, Gemma models served via Groq cloud.
"""

import time
from typing import Optional

from models.base_model import BaseModel, ModelResponse
from utils.logger import get_logger

logger = get_logger("system")


class GroqModel(BaseModel):
    """
    LLM provider for Groq cloud inference API.

    Supported models (as of 2026):
        - llama-3.1-8b-instant
        - llama-3.3-70b-versatile
    """

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name=model_name, provider="groq")
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Groq client."""
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "groq package not installed. Run: pip install groq"
                )
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ModelResponse:
        """Generate response using Groq API."""
        import asyncio

        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            client = self._get_client()

            # Groq client is synchronous — run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )

            latency = (time.time() - start_time) * 1000
            text = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else None

            logger.info(
                f"Groq [{self.model_name}] responded "
                f"({latency:.0f}ms, ~{tokens or '?'} tokens)"
            )

            return ModelResponse(
                text=text,
                model_name=self.model_name,
                provider="groq",
                tokens_used=tokens,
                latency_ms=latency,
                raw_response={"text": text},
            )

        except Exception as e:
            logger.error(f"Groq [{self.model_name}] error: {e}")
            raise ConnectionError(
                f"Failed to get response from Groq model '{self.model_name}': {e}"
            )

    async def is_available(self) -> bool:
        """Check if the Groq API is reachable with the provided key."""
        try:
            client = self._get_client()
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "Say ok"}],
                    max_tokens=5,
                ),
            )
            return bool(response.choices[0].message.content)
        except Exception:
            return False

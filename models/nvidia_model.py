"""
Council Mode - NVIDIA NIM Provider
Interfaces with models hosted on NVIDIA's NIM infrastructure.
"""

import time
import asyncio
from typing import Optional
from openai import OpenAI

from models.base_model import BaseModel, ModelResponse
from models.openai_model import OpenAICompatibleModel
from utils.logger import get_logger

logger = get_logger("models")

class NvidiaModel(OpenAICompatibleModel):
    """
    Provider for NVIDIA NIM API.
    Specifically supports 'thinking' models like Qwen 122B.
    """
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(
            model_name=model_name,
            provider="NVIDIA",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.6,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Generate response with NVIDIA-specific 'thinking' support."""
        start_time = time.time()
        
        messages = []
        # Mistral models can be sensitive to system prompts; merge if needed
        is_mistral = "mistral" in self.model_name.lower()
        
        if system_prompt:
            if is_mistral:
                prompt = f"{system_prompt}\n\nUser Question: {prompt}"
            else:
                messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Run in executor since openai client is synchronous
            loop = asyncio.get_event_loop()
            
            # NVIDIA reasoning models often use specific extra_body parameters
            extra_body = {}
            model_lower = self.model_name.lower()
            
            if "deepseek" in model_lower:
                # Optimized for speed: Medium effort provides the best balance for Flash models
                extra_body["chat_template_kwargs"] = {"thinking": True, "reasoning_effort": "medium"}
            elif "qwen" in model_lower or "gpt-oss" in model_lower:
                extra_body["chat_template_kwargs"] = {"enable_thinking": True}
            
            # Support for reasoning_effort in Mistral models
            if "mistral" in model_lower:
                extra_body["reasoning_effort"] = "high"

            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens if max_tokens > 4096 else 8192, # Increase for reasoning
                    extra_body=extra_body
                )
            )
            
            latency = (time.time() - start_time) * 1000
            
            # Extract content safely
            message = response.choices[0].message
            text = message.content or ""
            
            # Check for reasoning_content (common in NVIDIA NIM reasoning models)
            # Some models use 'reasoning', others use 'reasoning_content'
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
            
            # If content is empty but reasoning exists, try to extract the answer from reasoning
            if not text and reasoning:
                # Look for common answer headers in the reasoning trace (taking the LAST occurrence)
                import re
                # Find all sections starting with Answer or ## Answer
                answer_sections = list(re.finditer(r'(?i)(?:##\s*Answer|Answer)[:\s]*(.*)', reasoning, re.DOTALL))
                if answer_sections:
                    # Take the last section as the final answer
                    last_match = answer_sections[-1]
                    text = last_match.group(1).strip()
                    # Clean up the reasoning trace to remove the extracted answer
                    reasoning = reasoning[:last_match.start()].strip()
                else:
                    # Fallback: Use the whole reasoning if no header found
                    text = f"[Thinking Process Only]\n\n{reasoning}"

            return ModelResponse(
                text=text,
                model_name=self.model_name,
                provider=self.provider,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                latency_ms=latency,
                reasoning=reasoning,
                raw_response=response.model_dump()
            )
            
        except Exception as e:
            logger.error(f"Error from NVIDIA NIM model '{self.model_name}': {e}")
            raise ConnectionError(f"Failed to get response from NVIDIA NIM model '{self.model_name}': {e}")

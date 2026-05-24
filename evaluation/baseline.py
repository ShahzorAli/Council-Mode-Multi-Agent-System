"""
Council Mode - Single-Agent Baseline
Baseline evaluator that processes queries with a single LLM (no debate).
Used for comparative analysis against the Council (Multi-Agent) approach.
"""

import asyncio
import time
from typing import List
from dataclasses import dataclass

from models.base_model import BaseModel
from utils.logger import get_logger

logger = get_logger("system")


@dataclass
class BaselineResult:
    """Result from single-agent baseline."""
    query: str
    answer: str
    model_name: str
    latency_ms: float


class SingleAgentBaseline:
    """
    
    Processes queries with a single LLM (no RAG, no debate).
    This represents the "status quo" that Council Mode aims to improve upon.
    """
    
    def __init__(self, model: BaseModel):
        self.model = model
    
    async def process(self, query: str) -> BaselineResult:
        """Process a query with a single model."""
        start = time.time()
        
        response = await self.model.generate(
            prompt=query,
            system_prompt="Answer the following question accurately and concisely.",
            temperature=0.7,
        )
        
        latency = (time.time() - start) * 1000
        
        return BaselineResult(
            query=query,
            answer=response.text,
            model_name=str(self.model),
            latency_ms=latency,
        )
    
    async def batch_process(self, queries: List[str]) -> List[BaselineResult]:
        """Process multiple queries."""
        tasks = [self.process(q) for q in queries]
        return await asyncio.gather(*tasks)

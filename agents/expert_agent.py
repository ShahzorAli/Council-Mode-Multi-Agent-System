from typing import Optional

from models.base_model import BaseModel, ModelResponse
from utils.prompts import (
    EXPERT_SYSTEM_PROMPT,
    EXPERT_USER_TEMPLATE,
    EXPERT_REBUTTAL_PROMPT,
)
from utils.logger import get_logger

logger = get_logger("expert")


class ExpertResponse:
    """Structured response from an Expert Agent."""
    
    def __init__(
        self,
        expert_id: int,
        expert_name: str,
        model_name: str,
        answer: str,
        round_num: int = 1,
        raw_response: str = "",
        reasoning: Optional[str] = None,
    ):
        self.expert_id = expert_id
        self.expert_name = expert_name
        self.model_name = model_name
        self.answer = answer
        self.round_num = round_num
        self.raw_response = raw_response
        self.reasoning = reasoning
    
    def __repr__(self):
        return (
            f"ExpertResponse(expert={self.expert_name}, "
            f"model={self.model_name}, round={self.round_num})"
        )


class ExpertAgent:
    """
    Expert Agent for the Multi-Agent Debate protocol
    
    Each expert:
    - Receives the same RAG evidence
    - Generates an independent, citation-heavy analysis
    - Can be prompted to defend or correct claims in subsequent rounds
    
    """
    
    # Named personas for the three experts
    EXPERT_PERSONAS = {
        1: "The Analyst",
        2: "The Researcher",
        3: "The Specialist",
    }
    
    def __init__(self, expert_id: int, model: BaseModel):
        """
        Args:
            expert_id: Unique identifier (1, 2, or 3)
            model: The LLM model this expert uses
        """
        self.expert_id = expert_id
        self.model = model
        self.expert_name = self.EXPERT_PERSONAS.get(expert_id, f"Expert {expert_id}")
        self.response_history: list[ExpertResponse] = []
        
        logger.info(
            f"Expert Agent {expert_id} ('{self.expert_name}') "
            f"initialized with model: {model}"
        )
    
    async def generate_response(
        self,
        query: str,
        evidence: str,
    ) -> ExpertResponse:
        """
        Generate an initial evidence-based response (Round 1).
        
        Args:
            query: The user's original question
            evidence: Formatted evidence from RAG retrieval
            
        Returns:
            ExpertResponse with the expert's analysis
        """
        logger.info(f"Expert {self.expert_id} generating initial response...")
        
        system_prompt = EXPERT_SYSTEM_PROMPT.format(
            expert_id=self.expert_id,
            expert_name=self.expert_name,
        )
        
        user_prompt = EXPERT_USER_TEMPLATE.format(
            query=query,
            evidence=evidence,
        )
        
        try:
            response = await self.model.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=2048,
            )
            
            ans_text = response.text or ""
            expert_response = ExpertResponse(
                expert_id=self.expert_id,
                expert_name=self.expert_name,
                model_name=str(self.model),
                answer=ans_text,
                round_num=1,
                raw_response=ans_text,
                reasoning=response.reasoning,
            )
            
            self.response_history.append(expert_response)
            logger.info(f"Expert {self.expert_id} (Round 1) responded ({len(ans_text)} chars)")
            return expert_response
            
        except Exception as e:
            logger.error(f"Expert {self.expert_id} failed in Round 1: {e}")
            # Return a fallback response
            return ExpertResponse(
                expert_id=self.expert_id,
                expert_name=self.expert_name,
                model_name=str(self.model),
                answer=f"Error: Expert was unable to respond due to API limits ({str(e)})",
                round_num=1
            )
    
    async def generate_rebuttal(
        self,
        query: str,
        skeptic_critique: str,
        round_num: int,
        additional_evidence: str = "",
    ) -> Optional[ExpertResponse]:
        """
        Generate a rebuttal/correction response to the Skeptic's critique.
        
        """
        logger.info(f"Expert {self.expert_id} generating rebuttal (Round {round_num})...")
        
        # Get previous response
        previous_response = (
            self.response_history[-1].answer
            if self.response_history
            else "No previous response"
        )
        
        # --- TOKEN SAFETY TRUNCATION ---
        
        safe_critique = self._truncate_text(skeptic_critique, 4000)
        safe_prev = self._truncate_text(previous_response, 4000)
        safe_evidence = self._truncate_text(additional_evidence or "None", 6000)
        
        prompt = EXPERT_REBUTTAL_PROMPT.format(
            expert_id=self.expert_id,
            expert_name=self.expert_name,
            round_num=round_num,
            skeptic_critique=safe_critique,
            previous_response=safe_prev,
            additional_evidence=safe_evidence,
        )
        
        try:
            response = await self.model.generate(
                prompt=prompt,
                system_prompt="",
                temperature=0.3,
                max_tokens=2048,
            )
            
            expert_response = ExpertResponse(
                expert_id=self.expert_id,
                expert_name=self.expert_name,
                model_name=str(self.model),
                answer=response.text,
                round_num=round_num,
                raw_response=response.text,
                reasoning=response.reasoning,
            )
            
            self.response_history.append(expert_response)
            return expert_response
            
        except Exception as e:
            logger.error(f"Expert {self.expert_id} failed in Round {round_num}: {e}")
            # Return None to signify this expert failed this round
            return None

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Heuristic truncation to stay under token limits."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "... [truncated for length]"
    
    def get_latest_response(self) -> Optional[ExpertResponse]:
        """Get the most recent response from this expert."""
        return self.response_history[-1] if self.response_history else None
    
    def reset(self):
        """Clear response history for a new query."""
        self.response_history.clear()

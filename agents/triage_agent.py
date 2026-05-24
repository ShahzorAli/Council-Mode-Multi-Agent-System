import json
from typing import Optional

from models.base_model import BaseModel, ModelResponse
from utils.prompts import TRIAGE_SYSTEM_PROMPT, TRIAGE_USER_TEMPLATE
from utils.logger import get_logger

logger = get_logger("triage")


class TriageResult:
    """Result of query triage classification."""
    
    def __init__(
        self,
        classification: str,
        reason: str,
        confidence: float,
        raw_response: str = "",
    ):
        self.classification = classification.upper()
        self.reason = reason
        self.confidence = confidence
        self.raw_response = raw_response
    
    @property
    def is_high_stakes(self) -> bool:
        return self.classification == "HIGH_STAKES"
    
    @property
    def is_low_stakes(self) -> bool:
        return self.classification == "LOW_STAKES"
    
    def __repr__(self):
        return (
            f"TriageResult(classification={self.classification}, "
            f"confidence={self.confidence:.2f}, reason='{self.reason}')"
        )


class TriageAgent:
    """
    Intelligent Query Triage & Planning Agent (Module A).
    
    Uses a small, fast model to classify queries as:
    - LOW_STAKES: Route to direct single-model answer
    - HIGH_STAKES: Trigger the full Council debate pipeline

    """
    
    def __init__(self, model: BaseModel):
        """
        Args:
            model: A small, fast model for classification (e.g., Llama 3.1 8B)
        """
        self.model = model
        logger.info(f"Triage Agent initialized with model: {model}")
    
    async def classify(self, query: str) -> TriageResult:
        """
        Classify a user query by factual risk level.
        
        Args:
            query: The user's input query
            
        Returns:
            TriageResult with classification, reason, and confidence
        """
        logger.info(f"Classifying query: '{query[:80]}...'")
        
        prompt = TRIAGE_USER_TEMPLATE.format(query=query)
        
        try:
            response = await self.model.generate(
                prompt=prompt,
                system_prompt=TRIAGE_SYSTEM_PROMPT,
                temperature=0.1,  # Low temp for consistent classification
                max_tokens=256,
            )
            
            result = self._parse_response(response.text)
            
            logger.info(
                f"Query classified as {result.classification} "
                f"(confidence: {result.confidence:.2f}) — {result.reason}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Triage classification failed: {e}")
            # Default to HIGH_STAKES on failure (safer)
            return TriageResult(
                classification="HIGH_STAKES",
                reason=f"Classification failed ({e}), defaulting to high stakes",
                confidence=0.5,
            )
    
    def _parse_response(self, response_text: str) -> TriageResult:
        """Parse the model's JSON response into a TriageResult."""
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response (model might add extra text)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                return TriageResult(
                    classification=data.get("classification", "HIGH_STAKES"),
                    reason=data.get("reason", "No reason provided"),
                    confidence=float(data.get("confidence", 0.5)),
                    raw_response=response_text,
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # Fallback: heuristic parsing
        text_lower = response_text.lower()
        
        if "low_stakes" in text_lower or "low stakes" in text_lower:
            classification = "LOW_STAKES"
        else:
            classification = "HIGH_STAKES"
        
        return TriageResult(
            classification=classification,
            reason="Parsed from unstructured response",
            confidence=0.6,
            raw_response=response_text,
        )

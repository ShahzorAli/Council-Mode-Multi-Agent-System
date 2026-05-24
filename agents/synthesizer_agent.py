from typing import List
from models.base_model import BaseModel
from agents.expert_agent import ExpertResponse
from utils.prompts import SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_TEMPLATE
from utils.logger import get_logger

logger = get_logger("synthesizer")


class SynthesisResult:
    """Final synthesized output from the Council."""
    
    def __init__(self, final_answer, confidence_score, consensus_facts,
                 resolved_disputes, source_attribution, caveats, raw_response=""):
        self.final_answer = final_answer
        self.confidence_score = confidence_score
        self.consensus_facts = consensus_facts
        self.resolved_disputes = resolved_disputes
        self.source_attribution = source_attribution
        self.caveats = caveats
        self.raw_response = raw_response


class SynthesizerAgent:
    """
    Consensus Synthesis & Truth-Scoring Agent 
    
    Aggregates the full debate transcript and expert final positions
    into a single, verified response with confidence scores and citations.
    """
    
    def __init__(self, model: BaseModel):
        self.model = model
        logger.info(f"Synthesizer Agent initialized with model: {model}")
    
    async def synthesize(self, query, debate_transcript, final_expert_responses,
                         evidence, num_rounds):
        logger.info("Synthesizer generating final consensus...")
        
        final_positions = self._format_final_positions(final_expert_responses)
        
        prompt = SYNTHESIZER_USER_TEMPLATE.format(
            query=query, debate_transcript=debate_transcript,
            final_positions=final_positions, evidence=evidence,
        )
        
        try:
            response = await self.model.generate(
                prompt=prompt, system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
                temperature=0.2, max_tokens=3000,
            )
            
            result = self._parse_synthesis(response.text, num_rounds)
            logger.info(f"Synthesis complete. Confidence: {result.confidence_score}%")
            return result
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return SynthesisResult(
                final_answer=f"Error: Could not synthesize final answer due to API limits: {str(e)}",
                confidence_score=0,
                consensus_facts=[],
                resolved_disputes=[],
                source_attribution=[],
                caveats="Synthesis failed."
            )
    
    def _format_final_positions(self, responses):
        parts = []
        for r in responses:
            parts.append(
                f"--- Expert {r.expert_id} ({r.expert_name}) "
                f"[Round {r.round_num}] ---\n{r.answer}\n"
            )
        return "\n".join(parts)
    
    def _parse_synthesis(self, text, num_rounds):
        import re
        
        # Extract confidence score
        confidence = 70  # default
        conf_match = re.search(r'(\d+)%', text)
        if conf_match:
            confidence = int(conf_match.group(1))
        
        return SynthesisResult(
            final_answer=text,
            confidence_score=confidence,
            consensus_facts=self._extract_section_items(text, "Consensus Facts"),
            resolved_disputes=self._extract_section_items(text, "Resolved Disputes"),
            source_attribution=self._extract_section_items(text, "Source Attribution"),
            caveats=self._extract_section(text, "Caveats"),
            raw_response=text,
        )
    
    def _extract_section(self, text, header):
        import re
        pattern = rf"##\s*{re.escape(header)}\s*\n(.*?)(?=\n##|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _extract_section_items(self, text, header):
        import re
        section = self._extract_section(text, header)
        if not section:
            return []
        return [i.strip() for i in re.findall(r'[-*]\s*(.+)', section) if i.strip()]

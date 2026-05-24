import re
from typing import List, Optional

from models.base_model import BaseModel
from agents.expert_agent import ExpertResponse
from utils.prompts import SKEPTIC_SYSTEM_PROMPT, SKEPTIC_USER_TEMPLATE
from utils.logger import get_logger

logger = get_logger("skeptic")


class SkepticAnalysis:
    """Structured output from the Skeptic Agent's analysis."""
    
    def __init__(self, critique, contradictions, unsupported_claims,
                 logical_issues, missing_perspectives, needs_re_retrieval=False,
                 refined_queries=None, questions_for_redebate=None, round_num=1):
        self.critique = critique
        self.contradictions = contradictions
        self.unsupported_claims = unsupported_claims
        self.logical_issues = logical_issues
        self.missing_perspectives = missing_perspectives
        self.needs_re_retrieval = needs_re_retrieval
        self.refined_queries = refined_queries or []
        self.questions_for_redebate = questions_for_redebate or []
        self.round_num = round_num
    
    @property
    def has_issues(self):
        return bool(self.contradictions or self.unsupported_claims or self.logical_issues)
    
    @property
    def total_issues(self):
        return len(self.contradictions) + len(self.unsupported_claims) + len(self.logical_issues)


class SkepticAgent:
    """The Skeptic Agent """
    
    def __init__(self, model: BaseModel):
        self.model = model
        self.analysis_history = []
        logger.info(f"Skeptic Agent initialized with model: {model}")
    
    async def analyze(self, query, evidence, expert_responses, round_num=1):
        logger.info(f"Skeptic analyzing {len(expert_responses)} responses (Round {round_num})...")
        
        formatted = self._format_expert_responses(expert_responses)
        prompt = SKEPTIC_USER_TEMPLATE.format(query=query, evidence=evidence, expert_responses=formatted)
        
        try:
            response = await self.model.generate(
                prompt=prompt, system_prompt=SKEPTIC_SYSTEM_PROMPT,
                temperature=0.3, max_tokens=2048,
            )
            analysis = self._parse_analysis(response.text, round_num)
            self.analysis_history.append(analysis)
            logger.info(f"Skeptic: {len(analysis.contradictions)} contradictions, re-retrieve={analysis.needs_re_retrieval}")
            return analysis
        except Exception as e:
            logger.error(f"Skeptic analysis failed: {e}")
            # Return a "neutral" analysis so the debate can continue
            return SkepticAnalysis(
                critique=f"Analysis failed due to API error: {str(e)}",
                contradictions=[],
                unsupported_claims=[],
                logical_issues=[],
                missing_perspectives=[],
                round_num=round_num
            )
    
    def _format_expert_responses(self, responses):
        parts = []
        for r in responses:
            parts.append(f"--- Expert {r.expert_id} ({r.expert_name}) [{r.model_name}] ---\n{r.answer}\n")
        return "\n".join(parts)
    
    def _parse_analysis(self, text, round_num):
        contradictions = self._extract_items(text, "Contradictions Found")
        unsupported = self._extract_items(text, "Unsupported Claims")
        logical = self._extract_items(text, "Logical Issues")
        missing = self._extract_items(text, "Missing Perspectives")
        questions = self._extract_items(text, "Questions for Re-Debate")
        
        needs_re = False
        refined = []
        section = self._extract_section(text, "Needs Re-Retrieval")
        if section:
            needs_re = "yes" in section.lower()
            refined = re.findall(r'"([^"]+)"', section)
        
        return SkepticAnalysis(
            critique=text, contradictions=contradictions, unsupported_claims=unsupported,
            logical_issues=logical, missing_perspectives=missing, needs_re_retrieval=needs_re,
            refined_queries=refined, questions_for_redebate=questions, round_num=round_num,
        )
    
    def _extract_section(self, text, header):
        pattern = rf"##\s*{re.escape(header)}\s*\n(.*?)(?=\n##|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _extract_items(self, text, header):
        section = self._extract_section(text, header)
        if not section:
            return []
        return [i.strip() for i in re.findall(r'[-*]\s*(.+)', section) if i.strip()]
    
    def reset(self):
        self.analysis_history.clear()

"""
Council Mode - Debate Round Manager
Manages individual debate rounds: expert responses → skeptic analysis → rebuttals.
"""

import asyncio
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from agents.expert_agent import ExpertAgent, ExpertResponse
from agents.skeptic_agent import SkepticAgent, SkepticAnalysis
from agents.retrieval_agent import RetrievalAgent
from debate.consensus import ConsensusScore, calculate_consensus
from utils.logger import get_logger

logger = get_logger("debate")


@dataclass
class RoundResult:
    """Result of a single debate round."""
    round_num: int
    expert_responses: List[ExpertResponse]
    skeptic_analysis: Optional[SkepticAnalysis]
    consensus: ConsensusScore
    additional_evidence: str = ""
    transcript: str = ""


class RoundManager:
    """
    Manages individual rounds of the Multi-Agent Debate.
    
    Each round:
    1. All experts generate responses 
    2. Skeptic analyzes for contradictions
    3. If needed, retrieval agent fetches more evidence
    4. Experts generate rebuttals
    """
    
    def __init__(self, experts: List[ExpertAgent], skeptic: SkepticAgent,
                 retrieval_agent: RetrievalAgent):
        self.experts = experts
        self.skeptic = skeptic
        self.retrieval_agent = retrieval_agent
    
    async def run_initial_round(self, query: str, evidence: str) -> RoundResult:
        """Run the first debate round — all experts respond independently."""
        logger.info("=" * 60)
        logger.info("DEBATE ROUND 1 — Initial Expert Responses")
        logger.info("=" * 60)
        
        # SEQUENTIAL EXECUTION 
    
        expert_responses = []
        for expert in self.experts:
            try:
                response = await expert.generate_response(query, evidence)
                if response:
                    expert_responses.append(response)
                
                # The "Breathe" delay (3 seconds)
                if expert != self.experts[-1]:
                    await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"Expert {expert.expert_id} failed: {e}")
        
        # Check if any experts succeeded
        if not expert_responses:
            logger.error("All experts failed in Round 1.")
            return None
            
        # Skeptic analyzes
        skeptic_analysis = await self.skeptic.analyze(
            query, evidence, expert_responses, round_num=1
        )
        
        # Check for re-retrieval
        additional_evidence = ""
        if skeptic_analysis.needs_re_retrieval and skeptic_analysis.refined_queries:
            logger.info("Skeptic requested re-retrieval — fetching more evidence...")
            additional_evidence, _ = self.retrieval_agent.re_retrieve(
                skeptic_analysis.refined_queries
            )
        
        # Calculate consensus
        consensus = calculate_consensus(expert_responses, skeptic_analysis.total_issues, 1)
        
        # Build transcript
        transcript = self._build_transcript(expert_responses, skeptic_analysis, 1)
        
        return RoundResult(
            round_num=1,
            expert_responses=expert_responses,
            skeptic_analysis=skeptic_analysis,
            consensus=consensus,
            additional_evidence=additional_evidence,
            transcript=transcript,
        )
    
    async def run_rebuttal_round(self, query: str, round_num: int,
                                  skeptic_critique: str,
                                  additional_evidence: str = "") -> RoundResult:
        """Run a rebuttal round — experts defend or correct their claims."""
        logger.info("=" * 60)
        logger.info(f"DEBATE ROUND {round_num} — Expert Rebuttals")
        logger.info("=" * 60)
        
        # SEQUENTIAL EXECUTION (Frankenstein Strategy)
        expert_responses = []
        for expert in self.experts:
            try:
                response = await expert.generate_rebuttal(query, skeptic_critique, round_num, additional_evidence)
                if response:
                    expert_responses.append(response)
                
                # The "Breathe" delay
                if expert != self.experts[-1]:
                    await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"Expert {expert.expert_id} failed in Round {round_num}: {e}")
        
        if not expert_responses:
            logger.error(f"Round {round_num}: All experts failed to generate rebuttals.")
            return None
            
        # Updated evidence for skeptic
        current_evidence = self.retrieval_agent.all_evidence
        
        # Skeptic re-analyzes
        skeptic_analysis = await self.skeptic.analyze(
            query, current_evidence, expert_responses, round_num
        )
        
        # Check for additional re-retrieval
        new_evidence = ""
        if skeptic_analysis.needs_re_retrieval and skeptic_analysis.refined_queries:
            logger.info(f"Round {round_num}: Skeptic requested more evidence...")
            new_evidence, _ = self.retrieval_agent.re_retrieve(
                skeptic_analysis.refined_queries
            )
        
        consensus = calculate_consensus(expert_responses, skeptic_analysis.total_issues, round_num)
        transcript = self._build_transcript(expert_responses, skeptic_analysis, round_num)
        
        return RoundResult(
            round_num=round_num,
            expert_responses=expert_responses,
            skeptic_analysis=skeptic_analysis,
            consensus=consensus,
            additional_evidence=new_evidence,
            transcript=transcript,
        )
    
    def _build_transcript(self, responses, analysis, round_num):
        parts = [f"\n{'='*50}", f"ROUND {round_num} TRANSCRIPT", f"{'='*50}\n"]
        
        for r in responses:
            parts.append(f"[Expert {r.expert_id} - {r.expert_name}]")
            parts.append(r.answer)
            parts.append("")
        
        if analysis:
            parts.append(f"[SKEPTIC ANALYSIS]")
            parts.append(analysis.critique)
        
        return "\n".join(parts)

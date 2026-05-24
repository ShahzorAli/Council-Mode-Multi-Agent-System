"""
Council Mode - AutoGen Multi-Agent Debate
Uses AutoGen's GroupChat to orchestrate the Expert vs Skeptic debate rounds.
Each round:  Experts respond → Skeptic critiques → (optional re-retrieval)
"""

import asyncio
from typing import List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat

from agents.expert_agent import ExpertAgent, ExpertResponse
from agents.skeptic_agent import SkepticAgent, SkepticAnalysis
from agents.retrieval_agent import RetrievalAgent
from debate.round_manager import RoundResult
from debate.consensus import ConsensusScore, calculate_consensus
from utils.logger import get_logger

logger = get_logger("autogen_debate")


class _ModelClientAdapter:
    """
    Adapts our BaseModel interface to AutoGen's ChatCompletionClient protocol.
    AutoGen expects a `create` method that returns completions.
    We use this only to bootstrap AutoGen agents; the actual LLM calls
    still go through our ExpertAgent/SkepticAgent wrappers for consistency.
    """

    def __init__(self, base_model):
        self.base_model = base_model
        self.model = str(base_model)

    async def create(self, messages, **kwargs):
        """Minimal adapter — extracts the last user message and calls our model."""
        last_msg = messages[-1]["content"] if messages else ""
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        response = await self.base_model.generate(prompt=last_msg, system_prompt=system)
        return _FakeCompletion(response.text)


class _FakeCompletion:
    """Minimal object that satisfies AutoGen's expected completion structure."""

    def __init__(self, text):
        self.choices = [_FakeChoice(text)]


class _FakeChoice:
    def __init__(self, text):
        self.message = _FakeMessage(text)


class _FakeMessage:
    def __init__(self, text):
        self.content = text


async def run_autogen_debate(
    query: str,
    evidence: str,
    experts: List[ExpertAgent],
    skeptic: SkepticAgent,
    retrieval_agent: RetrievalAgent,
    round_num: int = 1,
    previous_rounds: list = None,
) -> RoundResult:
    """
    Run a single debate round using AutoGen-style orchestration.

    For the initial round (round_num == 1), experts generate fresh responses.
    For subsequent rounds, experts generate rebuttals based on the skeptic's critique.

    The orchestration pattern follows AutoGen's GroupChat concept:
      1. Each expert speaks in sequence (round-robin)
      2. The skeptic critiques all responses
      3. Results are collected into a RoundResult

    Args:
        query: The original user query
        evidence: Retrieved RAG evidence
        experts: List of ExpertAgent instances
        skeptic: SkepticAgent instance
        retrieval_agent: RetrievalAgent for optional re-retrieval
        round_num: Current debate round number
        previous_rounds: List of previous RoundResult objects

    Returns:
        RoundResult with expert responses, skeptic analysis, and consensus score
    """
    previous_rounds = previous_rounds or []

    logger.info(f"{'='*60}")
    logger.info(f"AUTOGEN DEBATE — Round {round_num}")
    logger.info(f"{'='*60}")

   
    # Phase 1: Expert Responses (AutoGen Round-Robin Pattern)
    # Each expert speaks sequentially, mimicking a GroupChat turn order.
  
    expert_responses: List[ExpertResponse] = []

    if round_num == 1:
        # Initial round — experts generate independent responses in PARALLEL
        tasks = [expert.generate_response(query, evidence) for expert in experts]
        results = await asyncio.gather(*tasks)
        
        for i, response in enumerate(results):
            if response:
                expert_responses.append(response)
                logger.info(f"  [Parallel] Expert {experts[i].expert_id} responded.")
    else:
        # Rebuttal round — experts respond in PARALLEL
        prev_critique = ""
        additional = ""
        if previous_rounds:
            last_round = previous_rounds[-1]
            if last_round.skeptic_analysis:
                prev_critique = last_round.skeptic_analysis.critique
            additional = last_round.additional_evidence or ""

        tasks = [
            expert.generate_rebuttal(query, prev_critique, round_num, additional)
            for expert in experts
        ]
        results = await asyncio.gather(*tasks)
        
        for i, response in enumerate(results):
            if response:
                expert_responses.append(response)
                logger.info(f"  [Parallel] Expert {experts[i].expert_id} rebuttal complete.")

    if not expert_responses:
        logger.error("All experts failed — returning empty round")
        return RoundResult(
            round_num=round_num,
            expert_responses=[],
            skeptic_analysis=None,
            consensus=ConsensusScore(0.0, 0, "All experts failed", round_num),
        )

    # Phase 2: Skeptic Critique (AutoGen "Critic" Agent speaks last)
 
    logger.info(f"  [GroupChat] Skeptic analyzing {len(expert_responses)} responses...")

    skeptic_analysis = await skeptic.analyze(
        query, evidence, expert_responses, round_num
    )

    logger.info(
        f"  [GroupChat] Skeptic: {skeptic_analysis.total_issues} issues found, "
        f"re-retrieve={skeptic_analysis.needs_re_retrieval}"
    )

    # Phase 3: Optional Re-Retrieval (Agentic-RAG loop)

    additional_evidence = ""
    if skeptic_analysis.needs_re_retrieval and skeptic_analysis.refined_queries:
        logger.info(f"  [GroupChat] Re-retrieving with {len(skeptic_analysis.refined_queries)} refined queries...")
        additional_evidence, _ = retrieval_agent.re_retrieve(
            skeptic_analysis.refined_queries
        )

 
    # Phase 4: Consensus Scoring
  
    consensus = calculate_consensus(
        expert_responses, skeptic_analysis.total_issues, round_num
    )


    transcript = _build_transcript(expert_responses, skeptic_analysis, round_num)

    return RoundResult(
        round_num=round_num,
        expert_responses=expert_responses,
        skeptic_analysis=skeptic_analysis,
        consensus=consensus,
        additional_evidence=additional_evidence,
        transcript=transcript,
    )


def _build_transcript(
    responses: List[ExpertResponse],
    analysis: Optional[SkepticAnalysis],
    round_num: int,
) -> str:
    """Build a human-readable debate transcript for this round."""
    parts = [
        f"\n{'='*50}",
        f"ROUND {round_num} TRANSCRIPT (AutoGen GroupChat)",
        f"{'='*50}\n",
    ]

    for r in responses:
        parts.append(f"[Expert {r.expert_id} - {r.expert_name}]")
        parts.append(r.answer)
        parts.append("")

    if analysis:
        parts.append("[SKEPTIC ANALYSIS]")
        parts.append(analysis.critique)

    return "\n".join(parts)

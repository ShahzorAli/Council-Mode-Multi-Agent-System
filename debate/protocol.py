"""
Council Mode - MAD Protocol (LangGraph + AutoGen)
The main Multi-Agent Debate orchestrator using LangGraph StateGraph
and AutoGen GroupChat for expert debate rounds.
"""

import asyncio
from typing import Optional
from dataclasses import dataclass, field

from agents.triage_agent import TriageAgent, TriageResult
from agents.expert_agent import ExpertAgent
from agents.skeptic_agent import SkepticAgent
from agents.synthesizer_agent import SynthesizerAgent, SynthesisResult
from agents.retrieval_agent import RetrievalAgent
from debate.graph import build_council_graph, CouncilState
from models.base_model import BaseModel
from utils.logger import get_logger

logger = get_logger("debate")


@dataclass
class CouncilResult:
    """Complete result from the Council Mode pipeline."""
    query: str
    triage: TriageResult
    was_debated: bool
    synthesis: Optional[SynthesisResult] = None
    direct_answer: Optional[str] = None
    baseline_answer: Optional[str] = None  # Zero-Shot Baseline (Original Intent)
    rounds: list = field(default_factory=list)
    total_rounds: int = 0
    full_transcript: str = ""


class CouncilProtocol:
    """
    The Multi-Agent Debate (MAD) Protocol — powered by LangGraph + AutoGen.

    Pipeline (LangGraph StateGraph):
    1. Triage Node — classifies query risk
    2. If LOW_STAKES → Direct Answer Node
    3. If HIGH_STAKES:
       a. Retrieve Node — RAG evidence fetching
       b. Debate Node (AutoGen GroupChat) — Expert round-robin + Skeptic critique
       c. Conditional loop back to Debate or forward to Synthesize
       d. Synthesize Node — final consensus answer
    """

    def __init__(
        self,
        triage_agent: TriageAgent,
        experts: list[ExpertAgent],
        skeptic: SkepticAgent,
        synthesizer: SynthesizerAgent,
        retrieval_agent: RetrievalAgent,
        direct_answer_model: BaseModel,
        max_rounds: int = 3,
        consensus_threshold: float = 0.7,
    ):
        self.triage = triage_agent
        self.experts = experts
        self.skeptic = skeptic
        self.synthesizer = synthesizer
        self.retrieval_agent = retrieval_agent
        self.direct_model = direct_answer_model
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold

        # Build the LangGraph compiled graph
        self.graph = build_council_graph(
            triage_agent=triage_agent,
            experts=experts,
            skeptic=skeptic,
            synthesizer=synthesizer,
            retrieval_agent=retrieval_agent,
            direct_answer_model=direct_answer_model,
            max_rounds=max_rounds,
        )

        logger.info(
            f"Council Protocol initialized (LangGraph + AutoGen): "
            f"{len(experts)} experts, max {max_rounds} rounds, "
            f"threshold {consensus_threshold}"
        )

    async def process_query(self, query: str, force_debate: bool = False) -> CouncilResult:
        """
        Process a query through the LangGraph Council Mode pipeline.

        Args:
            query: The user's question
            force_debate: Skip triage and force full debate

        Returns:
            CouncilResult with the complete output
        """
        logger.info(f"\n{'#'*60}")
        logger.info(f"COUNCIL MODE (LangGraph) — Processing Query")
        logger.info(f"Query: {query[:100]}...")
        logger.info(f"{'#'*60}\n")

        # Reset expert histories for a fresh query
        for expert in self.experts:
            expert.reset()
        self.skeptic.reset()

        # Prepare initial state
        initial_state: CouncilState = {
            "query": query,
            "force_debate": force_debate,
            "_max_rounds": self.max_rounds,
        }

        # Run the LangGraph pipeline
        final_state = await self.graph.ainvoke(initial_state)

        # Convert final state → CouncilResult
        return self._state_to_result(final_state)

    def _state_to_result(self, state: dict) -> CouncilResult:
        """Convert the LangGraph final state into a CouncilResult."""
        triage_result = state.get("triage_result")
        was_debated = state.get("synthesis") is not None

        return CouncilResult(
            query=state.get("query", ""),
            triage=triage_result,
            was_debated=was_debated,
            synthesis=state.get("synthesis"),
            direct_answer=state.get("direct_answer"),
            baseline_answer=state.get("baseline_answer"),
            rounds=state.get("debate_rounds", []),
            total_rounds=state.get("current_round", 0),
            full_transcript=state.get("debate_transcript", ""),
        )

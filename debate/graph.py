
import asyncio
from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END

from agents.triage_agent import TriageAgent, TriageResult
from agents.expert_agent import ExpertAgent
from agents.skeptic_agent import SkepticAgent
from agents.synthesizer_agent import SynthesizerAgent, SynthesisResult
from agents.retrieval_agent import RetrievalAgent
from debate.autogen_debate import run_autogen_debate
from models.base_model import BaseModel
from utils.prompts import DIRECT_ANSWER_PROMPT
from utils.logger import get_logger

logger = get_logger("langgraph")



class CouncilState(TypedDict, total=False):
    """Typed state that flows through the LangGraph pipeline."""
    # Input
    query: str
    force_debate: bool

    # Triage output
    triage_result: Optional[TriageResult]

    # Baseline (Original Intent)
    baseline_answer: Optional[str]

    # Low-stakes path
    direct_answer: Optional[str]

    # Retrieval
    evidence: str
    evidence_chunks: list

    # Debate
    debate_transcript: str
    debate_rounds: list          # list[RoundResult]
    current_round: int
    consensus_reached: bool

    # Synthesis
    synthesis: Optional[SynthesisResult]


# Node functions 


def make_triage_node(triage_agent: TriageAgent, direct_model: BaseModel):
    """Create the Triage node that classifies query risk AND generates baseline."""

    async def _triage(state: CouncilState) -> dict:
        query = state["query"]
        force = state.get("force_debate", False)

        # Classify
        if force:
            triage_result = TriageResult("HIGH_STAKES", "Forced debate mode", 1.0)
        else:
            triage_result = await triage_agent.classify(query)

        logger.info(f"Triage: {triage_result.classification} (confidence={triage_result.confidence:.2f})")

        # Generate zero-shot baseline (Original Intent)
        baseline_prompt = DIRECT_ANSWER_PROMPT.format(query=query)
        baseline_resp = await direct_model.generate(prompt=baseline_prompt, temperature=0.7)

        return {
            "triage_result": triage_result,
            "baseline_answer": baseline_resp.text,
        }

    return _triage


def make_direct_answer_node(direct_model: BaseModel):
    """Create the Direct Answer node for low-stakes queries."""

    async def _direct(state: CouncilState) -> dict:
        logger.info("LOW STAKES — Generating direct answer...")
        prompt = DIRECT_ANSWER_PROMPT.format(query=state["query"])
        response = await direct_model.generate(prompt=prompt, temperature=0.7)
        return {"direct_answer": response.text}

    return _direct


def make_retrieve_node(retrieval_agent: RetrievalAgent):
    """Create the Retrieval node that fetches RAG evidence."""

    async def _retrieve(state: CouncilState) -> dict:
        logger.info("Retrieving evidence from vector store...")
        retrieval_agent.reset()
        evidence, chunks = retrieval_agent.initial_retrieve(state["query"])
        if not chunks:
            logger.warning("No evidence found — debate will rely on model knowledge")
        return {
            "evidence": evidence,
            "evidence_chunks": chunks,
            "current_round": 0,
            "debate_rounds": [],
            "debate_transcript": "",
            "consensus_reached": False,
        }

    return _retrieve


def make_debate_node(
    experts: list,
    skeptic: SkepticAgent,
    retrieval_agent: RetrievalAgent,
):
    """Create the Debate node using AutoGen GroupChat."""

    async def _debate(state: CouncilState) -> dict:
        current_round = state.get("current_round", 0) + 1
        logger.info(f"DEBATE ROUND {current_round} — AutoGen GroupChat")

        round_result = await run_autogen_debate(
            query=state["query"],
            evidence=state.get("evidence", ""),
            experts=experts,
            skeptic=skeptic,
            retrieval_agent=retrieval_agent,
            round_num=current_round,
            previous_rounds=state.get("debate_rounds", []),
        )

        rounds = list(state.get("debate_rounds", []))
        rounds.append(round_result)

        transcript = state.get("debate_transcript", "")
        if round_result.transcript:
            transcript += "\n" + round_result.transcript

        # Update evidence if re-retrieval happened
        updated_evidence = retrieval_agent.all_evidence or state.get("evidence", "")

        return {
            "debate_rounds": rounds,
            "current_round": current_round,
            "debate_transcript": transcript,
            "consensus_reached": round_result.consensus.has_consensus,
            "evidence": updated_evidence,
        }

    return _debate


def make_synthesize_node(
    synthesizer: SynthesizerAgent,
    experts: list,
    retrieval_agent: RetrievalAgent,
):
    """Create the Synthesis node that produces the final consensus answer."""

    async def _synthesize(state: CouncilState) -> dict:
        logger.info("Synthesizing final consensus answer...")

        # Gather final expert positions
        final_responses = [e.get_latest_response() for e in experts]
        final_responses = [r for r in final_responses if r is not None]

        final_evidence = retrieval_agent.all_evidence or state.get("evidence", "")

        synthesis = await synthesizer.synthesize(
            query=state["query"],
            debate_transcript=state.get("debate_transcript", ""),
            final_expert_responses=final_responses,
            evidence=final_evidence,
            num_rounds=state.get("current_round", 1),
        )

        logger.info(f"COUNCIL DEBATE COMPLETE — Confidence: {synthesis.confidence_score}%")
        return {"synthesis": synthesis}

    return _synthesize



# Routing functions — conditional edges


def route_after_triage(state: CouncilState) -> str:
    """Route to direct_answer or retrieve based on triage classification."""
    triage = state.get("triage_result")
    if triage and triage.is_low_stakes:
        return "direct_answer"
    return "retrieve"


def route_after_debate(state: CouncilState) -> str:
    """Route to synthesize or loop back for another debate round."""
    if state.get("consensus_reached", False):
        logger.info("Consensus reached — moving to synthesis")
        return "synthesize"

    current = state.get("current_round", 1)
    max_rounds = state.get("_max_rounds", 3)

    if current >= max_rounds:
        logger.info(f"Max rounds ({max_rounds}) reached — moving to synthesis")
        return "synthesize"

    # Check if skeptic found no issues in the last round
    rounds = state.get("debate_rounds", [])
    if rounds:
        last = rounds[-1]
        if last.skeptic_analysis and not last.skeptic_analysis.has_issues:
            logger.info("Skeptic found no issues — moving to synthesis")
            return "synthesize"

    return "debate"


# Graph builder


def build_council_graph(
    triage_agent: TriageAgent,
    experts: list,
    skeptic: SkepticAgent,
    synthesizer: SynthesizerAgent,
    retrieval_agent: RetrievalAgent,
    direct_answer_model: BaseModel,
    max_rounds: int = 3,
) -> StateGraph:
    """
    Build the LangGraph StateGraph for Council Mode.

    Returns a compiled graph that accepts CouncilState and produces the final state.
    """

    # Create node functions
    triage_fn = make_triage_node(triage_agent, direct_answer_model)
    direct_fn = make_direct_answer_node(direct_answer_model)
    retrieve_fn = make_retrieve_node(retrieval_agent)
    debate_fn = make_debate_node(experts, skeptic, retrieval_agent)
    synthesize_fn = make_synthesize_node(synthesizer, experts, retrieval_agent)

    # Build the graph
    graph = StateGraph(CouncilState)

    # Add nodes
    graph.add_node("triage", triage_fn)
    graph.add_node("direct_answer", direct_fn)
    graph.add_node("retrieve", retrieve_fn)
    graph.add_node("debate", debate_fn)
    graph.add_node("synthesize", synthesize_fn)

    # Set entry point
    graph.set_entry_point("triage")

    # Conditional edge: triage → direct_answer | retrieve
    graph.add_conditional_edges("triage", route_after_triage, {
        "direct_answer": "direct_answer",
        "retrieve": "retrieve",
    })

    # retrieve → debate
    graph.add_edge("retrieve", "debate")

    # Conditional edge: debate → debate (loop) | synthesize
    graph.add_conditional_edges("debate", route_after_debate, {
        "debate": "debate",
        "synthesize": "synthesize",
    })

    # Terminal edges
    graph.add_edge("direct_answer", END)
    graph.add_edge("synthesize", END)

    return graph.compile()

"""
Council Mode - Consensus Scoring
Calculates agreement levels and confidence scores from debate results.
"""

from typing import List
from agents.expert_agent import ExpertResponse
from utils.logger import get_logger

logger = get_logger("debate")


class ConsensusScore:
    """Agreement analysis between expert responses."""
    
    def __init__(self, agreement_ratio, confidence_pct, details, round_num=1):
        self.agreement_ratio = agreement_ratio  # 0.0 to 1.0
        self.confidence_pct = confidence_pct    # 0 to 100
        self.details = details
        self.round_num = round_num
    
    @property
    def has_consensus(self):
        return self.agreement_ratio >= 0.7
    
    def __repr__(self):
        return f"ConsensusScore(agreement={self.agreement_ratio:.2f}, confidence={self.confidence_pct}%)"


def calculate_consensus(
    expert_responses: List[ExpertResponse],
    skeptic_issues_count: int = 0,
    round_num: int = 1,
    max_rounds: int = 3,
) -> ConsensusScore:
    """
    Calculate consensus score based on expert agreement and issue resolution.
    
    Factors:
    - Number of experts responding
    - Reduction in skeptic issues across rounds
   
    """
    num_experts = len(expert_responses)
    
    if num_experts == 0:
        return ConsensusScore(0.0, 0, "No expert responses", round_num)
    
    # Base agreement: starts at 0.5, increases with rounds
    round_factor = min(round_num / max_rounds, 1.0)
    
    # Issue penalty: more issues = lower agreement
    issue_penalty = min(skeptic_issues_count * 0.1, 0.4)
    
    # Calculate agreement ratio
    base = 0.5
    agreement = base + (round_factor * 0.3) - issue_penalty
    agreement = max(0.1, min(1.0, agreement))
    
    # Extract confidence from expert responses (heuristic)
    confidence_mentions = 0
    high_conf = 0
    for r in expert_responses:
        text = r.answer.upper()
        if "HIGH" in text and "CONFIDENCE" in text:
            high_conf += 1
            confidence_mentions += 1
        elif "MEDIUM" in text and "CONFIDENCE" in text:
            confidence_mentions += 1
        elif "LOW" in text and "CONFIDENCE" in text:
            confidence_mentions += 1
    
    # Expert confidence boost
    if confidence_mentions > 0:
        conf_ratio = high_conf / confidence_mentions
        agreement += conf_ratio * 0.2
        agreement = min(1.0, agreement)
    
    confidence_pct = int(agreement * 100)
    
    details = (
        f"Round {round_num}/{max_rounds}, "
        f"{num_experts} experts, "
        f"{skeptic_issues_count} remaining issues, "
        f"{high_conf}/{num_experts} high-confidence experts"
    )
    
    logger.info(f"Consensus: {agreement:.2f} ({confidence_pct}%) — {details}")
    
    return ConsensusScore(agreement, confidence_pct, details, round_num)

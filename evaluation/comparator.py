"""
Council Mode - Comparator
Compares Single-Agent vs Multi-Agent (Council) performance.
Placeholder for Phase 4 (HaluEval + DeepEval integration).
"""

from dataclasses import dataclass
from typing import List

from evaluation.baseline import BaselineResult
from debate.protocol import CouncilResult
from utils.logger import get_logger

logger = get_logger("system")


@dataclass 
class ComparisonResult:
    """Comparison between baseline and council results."""
    query: str
    baseline_answer: str
    council_answer: str
    baseline_latency_ms: float
    council_rounds: int
    council_confidence: int
    improvement_notes: str = ""


class Comparator:
    """
    Compares Single-Agent (baseline) vs Council Mode (multi-agent) results.

    """
    
    def compare(self, baseline: BaselineResult, council: CouncilResult) -> ComparisonResult:
        """Compare a baseline result against a council result."""
        
        council_answer = ""
        council_confidence = 0
        
        if council.synthesis:
            council_answer = council.synthesis.final_answer
            council_confidence = council.synthesis.confidence_score
        elif council.direct_answer:
            council_answer = council.direct_answer
            council_confidence = 100  # Low stakes = direct answer
        
        return ComparisonResult(
            query=baseline.query,
            baseline_answer=baseline.answer,
            council_answer=council_answer,
            baseline_latency_ms=baseline.latency_ms,
            council_rounds=council.total_rounds,
            council_confidence=council_confidence,
        )
    
    def batch_compare(self, baselines: List[BaselineResult],
                      councils: List[CouncilResult]) -> List[ComparisonResult]:
        """Compare lists of baseline and council results."""
        results = []
        for b, c in zip(baselines, councils):
            results.append(self.compare(b, c))
        return results
    
    def summary_report(self, comparisons: List[ComparisonResult]) -> str:
        """Generate a summary comparison report."""
        if not comparisons:
            return "No comparisons to report."
        
        avg_confidence = sum(c.council_confidence for c in comparisons) / len(comparisons)
        avg_rounds = sum(c.council_rounds for c in comparisons) / len(comparisons)
        
        report = [
            "=" * 60,
            "COUNCIL MODE vs SINGLE AGENT — Comparison Report",
            "=" * 60,
            f"Total queries: {len(comparisons)}",
            f"Average council confidence: {avg_confidence:.1f}%",
            f"Average debate rounds: {avg_rounds:.1f}",
            "",
            "Note: Full HaluEval/DeepEval metrics will be added in Phase 4.",
            "=" * 60,
        ]
        
        return "\n".join(report)

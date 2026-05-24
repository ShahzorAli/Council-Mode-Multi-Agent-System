"""
Council Mode - Benchmark Runner
Runs Council Mode and Single-Agent baseline on benchmark datasets,
collects metrics, and generates comparison reports.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from data.datasets.base_dataset import BaseDataset, BenchmarkSample
from data.datasets.registry import get_dataset, get_all_datasets, list_datasets
from evaluation.metrics import (
    score_single_sample, aggregate_scores,
    relative_reduction_hallucination,
)
from utils.logger import get_logger

logger = get_logger("benchmark")


@dataclass
class SampleResult:
    """Result of evaluating a single sample."""
    sample_id: str
    dataset_name: str
    question: str
    ground_truth: str
    
    # Answers
    baseline_answer: str = ""
    council_answer: str = ""
    
    # Scores
    baseline_scores: Dict[str, Any] = field(default_factory=dict)
    council_scores: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    baseline_latency_ms: float = 0.0
    council_latency_ms: float = 0.0
    council_rounds: int = 0
    council_confidence: int = 0
    was_debated: bool = False
    
    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "question": self.question[:200],
            "ground_truth": self.ground_truth[:200],
            "baseline_answer": self.baseline_answer[:500],
            "council_answer": self.council_answer[:500],
            "baseline_scores": self.baseline_scores,
            "council_scores": self.council_scores,
            "baseline_latency_ms": self.baseline_latency_ms,
            "council_latency_ms": self.council_latency_ms,
            "council_rounds": self.council_rounds,
            "council_confidence": self.council_confidence,
            "was_debated": self.was_debated,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report for a dataset."""
    dataset_name: str
    timestamp: str
    total_samples: int
    
    # Aggregate metrics
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    council_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Improvement analysis
    rrh: float = 0.0  # Relative Reduction in Hallucination
    f1_improvement: float = 0.0
    em_improvement: float = 0.0
    
    # Timing
    total_baseline_time_s: float = 0.0
    total_council_time_s: float = 0.0
    
    # Individual results
    sample_results: List[Dict] = field(default_factory=list)


class BenchmarkRunner:
    """
    Runs benchmark evaluations: Council Mode vs Single-Agent Baseline.
    
    Usage:
        runner = BenchmarkRunner(protocol, baseline_model)
        report = await runner.run_dataset("halueval", sample_limit=50)
        runner.save_report(report)
    """
    
    def __init__(self, protocol=None, baseline=None, results_dir: str = None):
        """
        Args:
            protocol: CouncilProtocol instance (for multi-agent evaluation)
            baseline: SingleAgentBaseline instance (for single-agent evaluation)
            results_dir: Directory to save benchmark results
        """
        self.protocol = protocol
        self.baseline = baseline
        
        from config import PROJECT_ROOT
        self.results_dir = Path(results_dir) if results_dir else PROJECT_ROOT / "data" / "benchmark_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_dataset(
        self,
        dataset_name: str,
        sample_limit: int = 50,
        run_baseline: bool = True,
        run_council: bool = True,
        progress_callback=None,
    ) -> BenchmarkReport:
        """
        Run benchmark on a specific dataset.
        
    
        """
        dataset = get_dataset(dataset_name, sample_limit=sample_limit)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BENCHMARK: {dataset.name}")
        logger.info(f"Samples: {len(dataset)} | Baseline: {run_baseline} | Council: {run_council}")
        logger.info(f"{'='*60}\n")
        
        sample_results = []
        total = len(dataset)
        
        for i, sample in enumerate(dataset):
            if progress_callback:
                progress_callback(i + 1, total, f"Processing: {sample.question[:50]}...")
            
            logger.info(f"\n--- Sample {i+1}/{total}: {sample.id} ---")
            result = await self._evaluate_sample(sample, run_baseline, run_council)
            sample_results.append(result)
        
        # Generate report
        report = self._generate_report(dataset.name, sample_results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BENCHMARK COMPLETE: {dataset.name}")
        logger.info(f"Baseline Avg F1: {report.baseline_metrics.get('avg_f1_score', 0):.4f}")
        logger.info(f"Council Avg F1:  {report.council_metrics.get('avg_f1_score', 0):.4f}")
        logger.info(f"F1 Improvement:  {report.f1_improvement:+.2f}%")
        logger.info(f"RRH:             {report.rrh:+.2f}%")
        logger.info(f"{'='*60}\n")
        
        return report
    
    async def run_all_datasets(
        self,
        sample_limit: int = 25,
        progress_callback=None,
    ) -> List[BenchmarkReport]:
        """Run benchmarks on all registered datasets."""
        reports = []
        dataset_names = list(list_datasets().keys())
        
        for ds_name in dataset_names:
            try:
                report = await self.run_dataset(
                    ds_name, sample_limit=sample_limit,
                    progress_callback=progress_callback,
                )
                reports.append(report)
            except Exception as e:
                logger.error(f"Failed to benchmark {ds_name}: {e}")
        
        return reports
    
    async def _evaluate_sample(
        self, sample: BenchmarkSample,
        run_baseline: bool, run_council: bool,
    ) -> SampleResult:
        """Evaluate a single benchmark sample."""
        result = SampleResult(
            sample_id=sample.id,
            dataset_name=sample.dataset_name,
            question=sample.question,
            ground_truth=sample.ground_truth,
        )
        
        # Build query — include context if available (simulates RAG)
        query = sample.question
        if sample.context:
            query = (
                f"Based on the following context, answer the question.\n\n"
                f"Context:\n{sample.context}\n\n"
                f"Question: {sample.question}"
            )
        
        # Run baseline
        if run_baseline and self.baseline:
            try:
                start = time.time()
                baseline_result = await self.baseline.process(query)
                result.baseline_latency_ms = (time.time() - start) * 1000
                result.baseline_answer = baseline_result.answer
                
                result.baseline_scores = score_single_sample(
                    prediction=baseline_result.answer,
                    ground_truth=sample.ground_truth,
                    context=sample.context,
                    is_answerable=sample.is_answerable,
                )
            except Exception as e:
                logger.error(f"Baseline failed for {sample.id}: {e}")
                result.baseline_answer = f"ERROR: {e}"
        
        # Run Council Mode
        if run_council and self.protocol:
            try:
                start = time.time()
                council_result = await self.protocol.process_query(
                    query, force_debate=True
                )
                result.council_latency_ms = (time.time() - start) * 1000
                
                if council_result.synthesis:
                    result.council_answer = council_result.synthesis.final_answer
                    result.council_confidence = council_result.synthesis.confidence_score
                elif council_result.direct_answer:
                    result.council_answer = council_result.direct_answer
                    result.council_confidence = 100
                
                result.council_rounds = council_result.total_rounds
                result.was_debated = council_result.was_debated
                
                result.council_scores = score_single_sample(
                    prediction=result.council_answer,
                    ground_truth=sample.ground_truth,
                    context=sample.context,
                    is_answerable=sample.is_answerable,
                )
            except Exception as e:
                logger.error(f"Council failed for {sample.id}: {e}")
                result.council_answer = f"ERROR: {e}"
        
        return result
    
    def _generate_report(self, dataset_name: str, results: List[SampleResult]) -> BenchmarkReport:
        """Generate aggregate report from sample results."""
        baseline_scores = [r.baseline_scores for r in results if r.baseline_scores]
        council_scores = [r.council_scores for r in results if r.council_scores]
        
        baseline_agg = aggregate_scores(baseline_scores) if baseline_scores else {}
        council_agg = aggregate_scores(council_scores) if council_scores else {}
        
        # Calculate improvements
        b_f1 = baseline_agg.get("avg_f1_score", 0)
        c_f1 = council_agg.get("avg_f1_score", 0)
        f1_imp = ((c_f1 - b_f1) / b_f1 * 100) if b_f1 > 0 else 0.0
        
        b_em = baseline_agg.get("avg_exact_match", 0)
        c_em = council_agg.get("avg_exact_match", 0)
        em_imp = ((c_em - b_em) / b_em * 100) if b_em > 0 else 0.0
        
        # RRH calculation
        b_hallu = baseline_agg.get("avg_hallucination_score", 0)
        c_hallu = council_agg.get("avg_hallucination_score", 0)
        rrh = relative_reduction_hallucination(b_hallu, c_hallu)
        
        return BenchmarkReport(
            dataset_name=dataset_name,
            timestamp=datetime.now().isoformat(),
            total_samples=len(results),
            baseline_metrics=baseline_agg,
            council_metrics=council_agg,
            rrh=rrh,
            f1_improvement=round(f1_imp, 2),
            em_improvement=round(em_imp, 2),
            total_baseline_time_s=sum(r.baseline_latency_ms for r in results) / 1000,
            total_council_time_s=sum(r.council_latency_ms for r in results) / 1000,
            sample_results=[r.to_dict() for r in results],
        )
    
    def save_report(self, report: BenchmarkReport) -> Path:
        """Save benchmark report to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{report.dataset_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        logger.info(f"Report saved: {filepath}")
        return filepath
    
    def print_report(self, report: BenchmarkReport):
        """Print a formatted benchmark report to console."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        
        # Header
        console.print(Panel(
            f"[bold]{report.dataset_name}[/bold]\n"
            f"Samples: {report.total_samples} | {report.timestamp}",
            title="Benchmark Report",
            border_style="cyan",
        ))
        
        # Metrics comparison table
        table = Table(title="Metrics Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Baseline (Single Agent)", style="red")
        table.add_column("Council (Multi-Agent)", style="green")
        table.add_column("Δ Change", style="yellow")
        
        metrics_to_show = [
            ("Avg F1 Score", "avg_f1_score"),
            ("Avg Exact Match", "avg_exact_match"),
            ("Answer Containment", "answer_containment_rate"),
            ("Avg Hallucination Score", "avg_hallucination_score"),
            ("Abstention F1", "abstention_f1"),
        ]
        
        for label, key in metrics_to_show:
            b_val = report.baseline_metrics.get(key)
            c_val = report.council_metrics.get(key)
            if b_val is not None or c_val is not None:
                b_str = f"{b_val:.4f}" if b_val is not None else "N/A"
                c_str = f"{c_val:.4f}" if c_val is not None else "N/A"
                if b_val and c_val:
                    delta = c_val - b_val
                    sign = "+" if delta >= 0 else ""
                    d_str = f"{sign}{delta:.4f}"
                else:
                    d_str = "—"
                table.add_row(label, b_str, c_str, d_str)
        
        console.print(table)
        
        # Summary stats
        summary = Table(title="Summary")
        summary.add_column("Stat", style="cyan")
        summary.add_column("Value", style="white")
        summary.add_row("F1 Improvement", f"{report.f1_improvement:+.2f}%")
        summary.add_row("EM Improvement", f"{report.em_improvement:+.2f}%")
        summary.add_row("RRH (Hallucination Reduction)", f"{report.rrh:+.2f}%")
        summary.add_row("Baseline Total Time", f"{report.total_baseline_time_s:.1f}s")
        summary.add_row("Council Total Time", f"{report.total_council_time_s:.1f}s")
        console.print(summary)

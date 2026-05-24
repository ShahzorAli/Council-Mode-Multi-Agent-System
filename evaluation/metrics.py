
import re
import string
from typing import List, Dict, Any, Optional
from collections import Counter
from utils.logger import get_logger

logger = get_logger("evaluation")



# Text Normalization 


def normalize_text(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    text = ' '.join(text.split())
    return text


def get_tokens(text: str) -> List[str]:
    """Tokenize normalized text."""
    return normalize_text(text).split()



# Core Metrics


def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization. Returns 1.0 or 0.0."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground truth."""
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_contains_ground_truth(prediction: str, ground_truth: str) -> bool:
    """Check if the prediction contains the ground truth answer."""
    return normalize_text(ground_truth) in normalize_text(prediction)



# Hallucination Detection Metrics (HaluEval)


def detect_hallucination_signals(answer: str, context: str) -> Dict[str, Any]:
    """
    Heuristic hallucination detection — checks for common signals.
    Returns a dict of signal flags and a hallucination score (0-1).
    """
    signals = {
        "makes_unsupported_claims": False,
        "contradicts_context": False,
        "uses_hedging": False,
        "mentions_uncertainty": False,
        "fabricates_specifics": False,
    }
    
    answer_lower = answer.lower()
    
    # Hedging language (good — agent knows it's uncertain)
    hedge_patterns = [
        "i'm not sure", "i don't know", "it's unclear",
        "the context doesn't", "cannot determine", "not enough information",
        "unanswerable", "no information provided",
    ]
    signals["uses_hedging"] = any(p in answer_lower for p in hedge_patterns)
    signals["mentions_uncertainty"] = any(
        p in answer_lower for p in ["may", "might", "possibly", "potentially", "uncertain"]
    )
    
    # Fabricated specifics (dates, numbers not in context)
    if context:
        context_lower = context.lower()
        # Find numbers in answer not in context
        answer_nums = set(re.findall(r'\b\d{4}\b', answer))
        context_nums = set(re.findall(r'\b\d{4}\b', context))
        fabricated_nums = answer_nums - context_nums
        signals["fabricates_specifics"] = len(fabricated_nums) > 0
    
    # Calculate hallucination score (lower = less likely hallucinated)
    score = 0.0
    if signals["fabricates_specifics"]:
        score += 0.4
    if not signals["uses_hedging"] and not context:
        score += 0.2
    if signals["mentions_uncertainty"]:
        score -= 0.1  # Good sign
    
    score = max(0.0, min(1.0, score))
    
    return {
        "signals": signals,
        "hallucination_score": score,
    }


def relative_reduction_hallucination(
    baseline_hallucination_rate: float,
    council_hallucination_rate: float,
) -> float:
    """
    Relative Reduction in Hallucination (RRH).
    RRH = (baseline_rate - council_rate) / baseline_rate * 100
    Higher is better. Returns percentage.
    """
    if baseline_hallucination_rate == 0:
        return 0.0
    rrh = ((baseline_hallucination_rate - council_hallucination_rate) 
           / baseline_hallucination_rate) * 100
    return round(rrh, 2)



# Abstention Metrics (SQuAD 2.0)


ABSTENTION_KEYWORDS = [
    "unanswerable", "cannot be answered", "not enough information",
    "i don't know", "cannot determine", "no answer",
    "the context does not", "the passage does not",
    "not mentioned", "insufficient information",
    "there is no information", "unable to answer",
]

def did_abstain(answer: str) -> bool:
    """Check if the model abstained from answering."""
    answer_lower = answer.lower()
    return any(kw in answer_lower for kw in ABSTENTION_KEYWORDS)


def abstention_accuracy(
    predictions: List[str],
    is_answerable: List[bool],
) -> Dict[str, float]:
    """
    Calculate abstention accuracy for unanswerable questions.
    Returns precision, recall, F1 for abstention detection.
    """
    tp = fp = fn = tn = 0
    
    for pred, answerable in zip(predictions, is_answerable):
        abstained = did_abstain(pred)
        should_abstain = not answerable
        
        if abstained and should_abstain:
            tp += 1  # Correctly abstained
        elif abstained and not should_abstain:
            fp += 1  # Wrongly abstained
        elif not abstained and should_abstain:
            fn += 1  # Should have abstained
        else:
            tn += 1  # Correctly answered
    
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    return {
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),
        "abstention_f1": round(f1, 4),
        "overall_accuracy": round(accuracy, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


# Aggregate Scoring


def score_single_sample(
    prediction: str,
    ground_truth: str,
    context: str = "",
    is_answerable: bool = True,
) -> Dict[str, Any]:
    """Score a single prediction against ground truth."""
    scores = {
        "exact_match": exact_match(prediction, ground_truth),
        "f1_score": f1_score(prediction, ground_truth),
        "contains_answer": answer_contains_ground_truth(prediction, ground_truth),
        "did_abstain": did_abstain(prediction),
        "is_answerable": is_answerable,
    }
    
    if context:
        hallu = detect_hallucination_signals(prediction, context)
        scores["hallucination_score"] = hallu["hallucination_score"]
        scores["hallucination_signals"] = hallu["signals"]
    
    # Correctness for unanswerable questions
    if not is_answerable:
        scores["abstention_correct"] = scores["did_abstain"]
    
    return scores


def aggregate_scores(all_scores: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate individual sample scores into dataset-level metrics."""
    if not all_scores:
        return {}
    
    n = len(all_scores)
    
    result = {
        "total_samples": n,
        "avg_exact_match": sum(s.get("exact_match", 0) for s in all_scores) / n,
        "avg_f1_score": sum(s.get("f1_score", 0) for s in all_scores) / n,
        "answer_containment_rate": sum(1 for s in all_scores if s.get("contains_answer")) / n,
    }
    
    # Hallucination scores (only if available)
    hallu_scores = [s["hallucination_score"] for s in all_scores if "hallucination_score" in s]
    if hallu_scores:
        result["avg_hallucination_score"] = sum(hallu_scores) / len(hallu_scores)
    
    # Abstention metrics
    answerable = [s for s in all_scores if s.get("is_answerable") is not None]
    if answerable:
        preds = ["abstain" if s["did_abstain"] else "answer" for s in answerable]
        is_ans = [s["is_answerable"] for s in answerable]
        abs_metrics = abstention_accuracy(
            ["unanswerable" if p == "abstain" else "some answer" for p in preds],
            is_ans,
        )
        result.update(abs_metrics)
    
    return result

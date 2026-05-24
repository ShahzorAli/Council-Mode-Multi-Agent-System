"""
Council Mode - HaluEval Dataset Loader
"""

import json
from pathlib import Path
from typing import List, Optional

from data.datasets.base_dataset import BaseDataset, BenchmarkSample
from config import PROJECT_ROOT

class HaluEvalDataset(BaseDataset):
    """Loader for the HaluEval QA dataset."""
    
    def __init__(self, sample_limit: Optional[int] = None):
        samples = self._load_samples(sample_limit)
        super().__init__("HaluEval", samples)
    
    def _load_samples(self, limit: Optional[int]) -> List[BenchmarkSample]:
        file_path = PROJECT_ROOT / "data" / "documents" / "HaluEval_QA.json"
        if not file_path.exists():
            return []
            
        with open(file_path, "r") as f:
            samples = []
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                
                try:
                    item = json.loads(line.strip())
                    samples.append(BenchmarkSample(
                        id=f"halueval_{i}",
                        question=item.get("question", ""),
                        ground_truth=item.get("right_answer", ""),  
                        dataset_name="HaluEval",
                        context=item.get("knowledge", "")
                    ))
                except json.JSONDecodeError:
                    continue
        return samples

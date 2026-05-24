"""
Council Mode - MultiHop-RAG Dataset Loader
"""

import json
from pathlib import Path
from typing import List, Optional

from data.datasets.base_dataset import BaseDataset, BenchmarkSample
from config import PROJECT_ROOT

class MultiHopRAGDataset(BaseDataset):
    """Loader for the MultiHop-RAG dataset."""
    
    def __init__(self, sample_limit: Optional[int] = None):
        samples = self._load_samples(sample_limit)
        super().__init__("MultiHopRAG", samples)
    
    def _load_samples(self, limit: Optional[int]) -> List[BenchmarkSample]:
        file_path = PROJECT_ROOT / "data" / "documents" / "MultiHopRAG.json"
        if not file_path.exists():
            return []
            
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Fallback: Try loading as JSONL
                f.seek(0)
                data = []
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
        samples = []
        for i, item in enumerate(data):
            if limit and i >= limit:
                break
                
            # Flatten evidence facts into a context
            context = "\n".join([e.get("fact", "") for e in item.get("evidence_list", [])])
                
            samples.append(BenchmarkSample(
                id=f"multihop_{i}",
                question=item.get("query", ""),
                ground_truth=item.get("answer", ""),
                dataset_name="MultiHopRAG",
                context=context
            ))
        return samples

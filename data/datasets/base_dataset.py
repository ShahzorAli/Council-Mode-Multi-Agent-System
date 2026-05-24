"""
Council Mode - Dataset Base Classes
Defines the structure for benchmark datasets.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator


@dataclass
class BenchmarkSample:
    """A single sample from a benchmark dataset."""
    id: str
    question: str
    ground_truth: str
    dataset_name: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_answerable: bool = True


class BaseDataset:
    """Base class for all benchmark datasets."""
    
    def __init__(self, name: str, samples: List[BenchmarkSample]):
        self.name = name
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[BenchmarkSample]:
        return iter(self.samples)
    
    def __getitem__(self, idx) -> BenchmarkSample:
        return self.samples[idx]

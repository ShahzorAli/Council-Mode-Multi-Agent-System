"""
Council Mode - Dataset Registry
Manages registration and loading of benchmark datasets.
"""

from typing import Dict, Type, Optional, List
from data.datasets.base_dataset import BaseDataset
from utils.logger import get_logger

logger = get_logger("datasets")

# Global registry of datasets
_REGISTRY: Dict[str, Type[BaseDataset]] = {}

def register_dataset(name: str):
    """Decorator to register a dataset class."""
    def decorator(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator

def get_dataset(name: str, **kwargs) -> BaseDataset:
    """Load a dataset by name."""
    name = name.lower()
    
    # Lazy imports to avoid circular dependencies
    from data.datasets.halueval import HaluEvalDataset
    from data.datasets.hotpotqa import HotpotQADataset
    from data.datasets.multihoprag import MultiHopRAGDataset
    
    if name == "halueval":
        return HaluEvalDataset(**kwargs)
    elif name == "hotpotqa":
        return HotpotQADataset(**kwargs)
    elif name == "multihoprag":
        return MultiHopRAGDataset(**kwargs)
    
    if name not in _REGISTRY:
        raise ValueError(f"Dataset '{name}' not found. Available: {list(_REGISTRY.keys())}")
    
    return _REGISTRY[name](**kwargs)

def list_datasets() -> Dict[str, str]:
    """List all available datasets."""
    # Hardcoded for now until full registry is implemented
    return {
        "halueval": "HaluEval QA dataset for hallucination detection",
        "hotpotqa": "HotpotQA dataset for multi-hop reasoning",
        "multihoprag": "MultiHop-RAG dataset for complex retrieval"
    }

def get_all_datasets(sample_limit: Optional[int] = None) -> List[BaseDataset]:
    """Get all registered datasets."""
    return [get_dataset(name, sample_limit=sample_limit) for name in list_datasets().keys()]

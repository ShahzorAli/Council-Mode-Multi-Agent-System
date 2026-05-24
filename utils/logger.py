import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()

# Module-specific log prefixes
MODULE_PREFIXES = {
    "triage": "[TRIAGE]",
    "retrieval": "[RAG]",
    "expert": "[EXPERT]",
    "skeptic": "[SKEPTIC]",
    "synthesizer": "[SYNTHESIZER]",
    "debate": "[DEBATE]",
    "system": "[SYSTEM]",
    "evaluation": "[EVAL]",
    "benchmark": "[BENCHMARK]",
    "datasets": "[DATASETS]",
}


def get_logger(module_name: str) -> logging.Logger:
  
    prefix = MODULE_PREFIXES.get(module_name, f"[{module_name.upper()}]")
    
    logger = logging.getLogger(f"council.{module_name}")
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        # Simple format: YYYY-MM-DD HH:MM:SS [MODULE] MESSAGE
        formatter = logging.Formatter(f"%(asctime)s {prefix:13} %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def set_log_level(level: str):
    """Set log level for all council loggers."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    for name in MODULE_PREFIXES:
        logging.getLogger(f"council.{name}").setLevel(numeric_level)

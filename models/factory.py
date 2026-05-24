"""
Council Mode - Model Factory
Creates appropriate model instances based on configuration strings.

"""

from models.base_model import BaseModel
from config import (
    ollama_config, gemini_config, groq_config,
    openrouter_config, deepseek_config, cerebras_config,
    nvidia_config
)
from utils.logger import get_logger

logger = get_logger("system")


def create_model(model_string: str) -> BaseModel:
    """
    Create a model instance from a config string.

    """
    from models.openai_model import OpenAICompatibleModel
    
    parts = model_string.split(":", 1)

    if len(parts) < 2:
        raise ValueError(
            f"Invalid model string '{model_string}'. "
            f"Expected format: 'provider:model_name'"
        )

    provider = parts[0].lower()
    model_name = parts[1]

    if provider == "groq":
        from models.groq_model import GroqModel
        if not groq_config.api_key:
            raise ValueError("GROQ_API_KEY not set.")
        logger.info(f"Creating Groq model: {model_name}")
        return GroqModel(model_name=model_name, api_key=groq_config.api_key)

    elif provider == "gemini":
        from models.gemini_model import GeminiModel
        if not gemini_config.api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        logger.info(f"Creating Gemini model: {model_name}")
        return GeminiModel(model_name=model_name, api_key=gemini_config.api_key)

    elif provider == "ollama":
        from models.ollama_model import OllamaModel
        logger.info(f"Creating Ollama model: {model_name}")
        return OllamaModel(model_name=model_name, base_url=ollama_config.base_url)
        
    elif provider == "openrouter":
        if not openrouter_config.api_key:
            raise ValueError("OPENROUTER_API_KEY not set.")
        logger.info(f"Creating OpenRouter model: {model_name}")
        return OpenAICompatibleModel(
            model_name=model_name,
            provider="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_config.api_key
        )
        
    elif provider == "deepseek":
        if not deepseek_config.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set.")
        logger.info(f"Creating DeepSeek model: {model_name}")
        return OpenAICompatibleModel(
            model_name=model_name,
            provider="DeepSeek",
            base_url="https://api.deepseek.com",
            api_key=deepseek_config.api_key
        )
        
    elif provider == "cerebras":
        if not cerebras_config.api_key:
            raise ValueError("CEREBRAS_API_KEY not set.")
        logger.info(f"Creating Cerebras model: {model_name}")
        return OpenAICompatibleModel(
            model_name=model_name,
            provider="Cerebras",
            base_url="https://api.cerebras.ai/v1",
            api_key=cerebras_config.api_key
        )
        
    elif provider == "nvidia":
        from models.nvidia_model import NvidiaModel
        if not nvidia_config.api_key:
            raise ValueError("NVIDIA_API_KEY not set.")
        logger.info(f"Creating NVIDIA model: {model_name}")
        return NvidiaModel(model_name=model_name, api_key=nvidia_config.api_key)

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: 'groq', 'gemini', 'ollama', 'openrouter', 'deepseek', 'cerebras'"
        )

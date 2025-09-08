"""
OpenAI embedding strategy.
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from .base import EmbeddingStrategy
from ..config import EmbeddingConfig


class OpenAIEmbeddingStrategy(EmbeddingStrategy):
    """Strategy for OpenAI embeddings."""
    
    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        """Create OpenAI embeddings."""
        params = config.params.copy()
        return OpenAIEmbeddings(
            model=config.model_name,
            **params
        )
    
    def validate_config(self, config: EmbeddingConfig) -> None:
        """Validate OpenAI embedding configuration."""
        # Check for API key
        api_key = config.params.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI embeddings require OPENAI_API_KEY environment variable or api_key in config")
        
        # Validate model name
        valid_models = [
            'text-embedding-3-small', 
            'text-embedding-3-large', 
            'text-embedding-ada-002'
        ]
        if config.model_name not in valid_models:
            raise ValueError(f"Unsupported OpenAI model: {config.model_name}. Supported: {valid_models}") 
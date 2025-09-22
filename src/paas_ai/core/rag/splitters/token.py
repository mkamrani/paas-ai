"""
Token text splitter strategy.
"""

from langchain_text_splitters import TokenTextSplitter

from .base import SplitterStrategy
from ..config import SplitterConfig

# Import tiktoken at module level for better testability
try:
    import tiktoken
except ImportError:
    tiktoken = None


class TokenSplitterStrategy(SplitterStrategy):
    """Strategy for token-based text splitting."""
    
    def create_splitter(self, config: SplitterConfig):
        """Create a token text splitter."""
        params = config.params.copy()
        return TokenTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            **params
        )
    
    def validate_config(self, config: SplitterConfig) -> None:
        """Validate token splitter configuration."""
        if config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if config.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
            
        # Validate encoding if provided
        encoding_name = config.params.get('encoding_name')
        if encoding_name is not None:
            if tiktoken is None:
                raise ValueError("tiktoken is required for encoding validation but not installed")
            try:
                tiktoken.get_encoding(encoding_name)
            except Exception:
                raise ValueError(f"Invalid encoding: {encoding_name}")
            
        # Validate model_name if provided  
        model_name = config.params.get('model_name')
        if model_name is not None:
            if not isinstance(model_name, str):
                raise ValueError("model_name must be a string") 
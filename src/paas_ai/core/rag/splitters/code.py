"""
Code text splitter strategy.
"""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter as CodeSplitter,
    Language
)

from .base import SplitterStrategy
from ..config import SplitterConfig


class CodeSplitterStrategy(SplitterStrategy):
    """Strategy for programming language-aware text splitting."""
    
    def create_splitter(self, config: SplitterConfig):
        """Create a code-aware text splitter."""
        params = config.params.copy()
        language = params.get('language', Language.PYTHON)
        
        # Ensure language is a Language enum
        if isinstance(language, str):
            try:
                language = Language(language.lower())
            except ValueError:
                raise ValueError(f"Unsupported language: {language}")
        
        return CodeSplitter.from_language(
            language=language,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            **{k: v for k, v in params.items() if k != 'language'}
        )
    
    def validate_config(self, config: SplitterConfig) -> None:
        """Validate code splitter configuration."""
        if config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if config.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
            
        # Validate language if provided
        language = config.params.get('language')
        if language is not None:
            if isinstance(language, str):
                # Check if it's a valid Language enum value
                valid_languages = [lang.value for lang in Language]
                if language.lower() not in valid_languages:
                    raise ValueError(f"Unsupported language: {language}. Supported: {valid_languages}")
            elif not isinstance(language, Language):
                raise ValueError("language must be a string or Language enum") 
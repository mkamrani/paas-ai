"""
Strategy registry for text splitters.
"""

from typing import Dict, Type
from .base import SplitterStrategy
from ..config import SplitterType

# Import all strategies
from .character import CharacterSplitterStrategy
from .recursive_character import RecursiveCharacterSplitterStrategy
from .markdown import MarkdownSplitterStrategy
from .html import HTMLSplitterStrategy
from .json import JSONSplitterStrategy
from .code import CodeSplitterStrategy
from .token import TokenSplitterStrategy


class SplitterRegistry:
    """Registry for text splitter strategies."""
    
    _strategies: Dict[SplitterType, Type[SplitterStrategy]] = {
        SplitterType.CHARACTER: CharacterSplitterStrategy,
        SplitterType.RECURSIVE_CHARACTER: RecursiveCharacterSplitterStrategy,
        SplitterType.MARKDOWN: MarkdownSplitterStrategy,
        SplitterType.HTML: HTMLSplitterStrategy,
        SplitterType.JSON: JSONSplitterStrategy,
        SplitterType.CODE: CodeSplitterStrategy,
        SplitterType.TOKEN: TokenSplitterStrategy,
    }
    
    @classmethod
    def get_strategy(cls, splitter_type: SplitterType) -> SplitterStrategy:
        """Get strategy instance for the given splitter type."""
        if splitter_type not in cls._strategies:
            raise ValueError(f"No strategy registered for splitter type: {splitter_type}")
        
        strategy_class = cls._strategies[splitter_type]
        return strategy_class()
    
    @classmethod
    def register_strategy(cls, splitter_type: SplitterType, strategy_class: Type[SplitterStrategy]) -> None:
        """Register a new strategy for a splitter type."""
        cls._strategies[splitter_type] = strategy_class
    
    @classmethod
    def list_supported_types(cls) -> list[SplitterType]:
        """List all supported splitter types."""
        return list(cls._strategies.keys()) 
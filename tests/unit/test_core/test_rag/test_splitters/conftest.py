"""
Shared fixtures for splitter tests.
"""

import pytest
from src.paas_ai.core.rag.splitters.registry import SplitterRegistry
from src.paas_ai.core.rag.config import SplitterType

# Import all strategies to ensure they're available
from src.paas_ai.core.rag.splitters.character import CharacterSplitterStrategy
from src.paas_ai.core.rag.splitters.recursive_character import RecursiveCharacterSplitterStrategy
from src.paas_ai.core.rag.splitters.markdown import MarkdownSplitterStrategy
from src.paas_ai.core.rag.splitters.html import HTMLSplitterStrategy
from src.paas_ai.core.rag.splitters.json import JSONSplitterStrategy
from src.paas_ai.core.rag.splitters.code import CodeSplitterStrategy
from src.paas_ai.core.rag.splitters.token import TokenSplitterStrategy


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry to its original state before and after each test."""
    # Store the original strategies
    original_strategies = {
        SplitterType.CHARACTER: CharacterSplitterStrategy,
        SplitterType.RECURSIVE_CHARACTER: RecursiveCharacterSplitterStrategy,
        SplitterType.MARKDOWN: MarkdownSplitterStrategy,
        SplitterType.HTML: HTMLSplitterStrategy,
        SplitterType.JSON: JSONSplitterStrategy,
        SplitterType.CODE: CodeSplitterStrategy,
        SplitterType.TOKEN: TokenSplitterStrategy,
    }
    
    # Store original methods
    original_get_strategy = SplitterRegistry.get_strategy
    original_register_strategy = SplitterRegistry.register_strategy
    original_list_supported_types = SplitterRegistry.list_supported_types
    
    yield
    
    # Reset everything to original state
    SplitterRegistry._strategies = original_strategies.copy()
    SplitterRegistry.get_strategy = original_get_strategy
    SplitterRegistry.register_strategy = original_register_strategy
    SplitterRegistry.list_supported_types = original_list_supported_types

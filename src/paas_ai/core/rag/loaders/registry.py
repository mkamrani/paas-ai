"""
Strategy registry for document loaders.
"""

from typing import Dict, Type
from .base import LoaderStrategy
from ..config import LoaderType

# Import all strategies
from .web import WebLoaderStrategy
from .crawl4ai_web import Crawl4AIWebLoaderStrategy
from .html import HTMLLoaderStrategy
from .pdf import PDFLoaderStrategy  
from .markdown import MarkdownLoaderStrategy
from .json import JSONLoaderStrategy
from .csv import CSVLoaderStrategy
from .directory import DirectoryLoaderStrategy
from .confluence import ConfluenceLoaderStrategy
from .notion import NotionLoaderStrategy
from .github import GitHubLoaderStrategy


class LoaderRegistry:
    """Registry for document loader strategies."""
    
    _strategies: Dict[LoaderType, Type[LoaderStrategy]] = {
        LoaderType.WEB: WebLoaderStrategy,
        LoaderType.CRAWL4AI_WEB: Crawl4AIWebLoaderStrategy,
        LoaderType.HTML: HTMLLoaderStrategy,
        LoaderType.PDF: PDFLoaderStrategy,
        LoaderType.MARKDOWN: MarkdownLoaderStrategy,
        LoaderType.JSON: JSONLoaderStrategy,
        LoaderType.CSV: CSVLoaderStrategy,
        LoaderType.DIRECTORY: DirectoryLoaderStrategy,
        LoaderType.CONFLUENCE: ConfluenceLoaderStrategy,
        LoaderType.NOTION: NotionLoaderStrategy,
        LoaderType.GITHUB: GitHubLoaderStrategy,
    }
    
    @classmethod
    def get_strategy(cls, loader_type: LoaderType) -> LoaderStrategy:
        """Get strategy instance for the given loader type."""
        if loader_type not in cls._strategies:
            raise ValueError(f"No strategy registered for loader type: {loader_type}")
        
        strategy_class = cls._strategies[loader_type]
        return strategy_class()
    
    @classmethod
    def register_strategy(cls, loader_type: LoaderType, strategy_class: Type[LoaderStrategy]) -> None:
        """Register a new strategy for a loader type."""
        cls._strategies[loader_type] = strategy_class
    
    @classmethod
    def list_supported_types(cls) -> list[LoaderType]:
        """List all supported loader types."""
        return list(cls._strategies.keys()) 
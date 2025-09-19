"""
Document loaders for RAG system.
"""

from .factory import DocumentLoaderFactory
from .registry import LoaderRegistry
from .base import LoaderStrategy

# Import all strategies for convenience
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

__all__ = [
    'DocumentLoaderFactory',
    'LoaderRegistry', 
    'LoaderStrategy',
    'WebLoaderStrategy',
    'Crawl4AIWebLoaderStrategy',
    'HTMLLoaderStrategy',
    'PDFLoaderStrategy',
    'MarkdownLoaderStrategy',
    'JSONLoaderStrategy',
    'CSVLoaderStrategy',
    'DirectoryLoaderStrategy',
    'ConfluenceLoaderStrategy',
    'NotionLoaderStrategy',
    'GitHubLoaderStrategy',
] 
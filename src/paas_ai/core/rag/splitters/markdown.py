"""
Markdown text splitter strategy.
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter

from .base import SplitterStrategy
from ..config import SplitterConfig


class MarkdownSplitterStrategy(SplitterStrategy):
    """Strategy for markdown-aware text splitting."""
    
    def create_splitter(self, config: SplitterConfig):
        """Create a markdown header text splitter."""
        params = config.params.copy()
        headers_to_split_on = params.get('headers_to_split_on', [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ])
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            **{k: v for k, v in params.items() if k != 'headers_to_split_on'}
        )
    
    def validate_config(self, config: SplitterConfig) -> None:
        """Validate markdown splitter configuration."""
        headers = config.params.get('headers_to_split_on')
        if headers is not None:
            if not isinstance(headers, list):
                raise ValueError("headers_to_split_on must be a list of tuples")
            for header in headers:
                if not isinstance(header, (list, tuple)) or len(header) != 2:
                    raise ValueError("each header must be a tuple of (markdown_header, name)")
                if not isinstance(header[0], str) or not isinstance(header[1], str):
                    raise ValueError("header tuple must contain two strings") 
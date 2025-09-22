"""
HTML text splitter strategy.
"""

from langchain_text_splitters import HTMLHeaderTextSplitter

from .base import SplitterStrategy
from ..config import SplitterConfig


class HTMLSplitterStrategy(SplitterStrategy):
    """Strategy for HTML-aware text splitting."""
    
    def create_splitter(self, config: SplitterConfig):
        """Create an HTML header text splitter."""
        params = config.params.copy()
        headers_to_split_on = params.get('headers_to_split_on', [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
        ])
        return HTMLHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            **{k: v for k, v in params.items() if k != 'headers_to_split_on'}
        )
    
    def validate_config(self, config: SplitterConfig) -> None:
        """Validate HTML splitter configuration."""
        headers = config.params.get('headers_to_split_on')
        if headers is not None:
            if not isinstance(headers, list):
                raise ValueError("headers_to_split_on must be a list of tuples")
            for header in headers:
                if not isinstance(header, (list, tuple)) or len(header) != 2:
                    raise ValueError("each header must be a tuple of (html_tag, name)")
                if not isinstance(header[0], str) or not isinstance(header[1], str):
                    raise ValueError("header tuple must contain two strings")
                # Validate HTML tag format
                tag = header[0].lower()
                if not tag.startswith('h') or not tag[1:].isdigit():
                    raise ValueError(f"HTML header tag must be h1-h6, got: {header[0]}")
                # Validate that the number is between 1-6
                tag_number = int(tag[1:])
                if tag_number < 1 or tag_number > 6:
                    raise ValueError(f"HTML header tag must be h1-h6, got: {header[0]}") 
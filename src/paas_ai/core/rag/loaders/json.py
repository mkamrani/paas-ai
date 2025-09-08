"""
JSON document loader strategy.
"""

import os
import json
from typing import Dict, Any
from langchain_community.document_loaders import JSONLoader
from langchain_core.document_loaders import BaseLoader

from .base import LoaderStrategy
from ..config import LoaderConfig


class JSONLoaderStrategy(LoaderStrategy):
    """Strategy for loading JSON documents."""
    
    def create_loader(self, config: LoaderConfig, url: str) -> BaseLoader:
        """Create a JSON document loader."""
        params = config.params.copy()
        return JSONLoader(
            file_path=url,
            jq_schema=params.get('jq_schema', '.'),
            text_content=params.get('text_content', False),
            **{k: v for k, v in params.items() if k not in ['jq_schema', 'text_content']}
        )
    
    def validate_config(self, config: LoaderConfig, url: str) -> None:
        """Validate JSON loader configuration."""
        # Check if it's a local file
        if not url.startswith(('http://', 'https://')):
            if not os.path.exists(url):
                raise ValueError(f"JSON file not found: {url}")
            if not url.lower().endswith('.json'):
                raise ValueError(f"File does not have .json extension: {url}")
                
            # Validate JSON syntax if it's a local file
            try:
                with open(url, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {url} - {e}")
        elif not url.lower().endswith('.json'):
            raise ValueError(f"URL does not point to a JSON file: {url}") 
"""
Strategy for Crawl4AI web loader.
"""

import logging
from typing import Dict, Any
from langchain_core.document_loaders import BaseLoader

from .base import LoaderStrategy
from .crawl4ai_web_loader import Crawl4AIWebLoader
from ..config import LoaderConfig

logger = logging.getLogger(__name__)


class Crawl4AIWebLoaderStrategy(LoaderStrategy):
    """Strategy for creating Crawl4AI web loaders."""
    
    def create_loader(self, config: LoaderConfig, url: str) -> BaseLoader:
        """Create a Crawl4AI web loader."""
        # Extract configuration parameters
        loader_config = config.params if hasattr(config, 'params') and config.params is not None else {}
        
        # Create and return the Crawl4AI web loader
        return Crawl4AIWebLoader(
            web_paths=[url],
            headless=loader_config.get('headless', True),
            wait_time=loader_config.get('wait_time', 3.0),
            timeout=loader_config.get('timeout', 30000),
            enable_stealth=loader_config.get('enable_stealth', False)  # Disable stealth by default due to import issues
        )
    
    def validate_config(self, config: LoaderConfig, url: str) -> None:
        """Validate configuration for Crawl4AI web loader."""
        if not url:
            raise ValueError("URL is required for Crawl4AI web loader")
        
        if not (url.startswith(('http://', 'https://')) or url.endswith('.csv')):
            raise ValueError(f"Invalid URL format for Crawl4AI web loader: {url}")
        
        # Validate configuration parameters if present
        loader_config = config.params if hasattr(config, 'params') and config.params is not None else {}
        
        if 'timeout' in loader_config:
            timeout = loader_config['timeout']
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValueError(f"Invalid timeout value: {timeout}")
        
        if 'wait_time' in loader_config:
            wait_time = loader_config['wait_time']
            if not isinstance(wait_time, (int, float)) or wait_time < 0:
                raise ValueError(f"Invalid wait_time value: {wait_time}")
        
        logger.info(f"Validated Crawl4AI web loader config for URL: {url}") 
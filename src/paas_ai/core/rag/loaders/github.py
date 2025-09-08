"""
GitHub document loader strategy.
"""

import os
from typing import Dict, Any
from langchain_community.document_loaders import GitHubIssuesLoader
from langchain_core.document_loaders import BaseLoader

from .base import LoaderStrategy
from ..config import LoaderConfig


class GitHubLoaderStrategy(LoaderStrategy):
    """Strategy for loading GitHub documents."""
    
    def create_loader(self, config: LoaderConfig, url: str) -> BaseLoader:
        """Create a GitHub document loader."""
        params = config.params.copy()
        
        # Extract repo info from GitHub URL
        if 'github.com' not in url:
            raise ValueError(f"Invalid GitHub URL format: {url}")
            
        parts = url.split('/')
        if len(parts) < 5:
            raise ValueError(f"Invalid GitHub URL format: {url}")
            
        repo = f"{parts[-2]}/{parts[-1]}"
        return GitHubIssuesLoader(
            repo=repo,
            access_token=params.get('access_token', os.getenv('GITHUB_TOKEN')),
            **{k: v for k, v in params.items() if k != 'access_token'}
        )
    
    def validate_config(self, config: LoaderConfig, url: str) -> None:
        """Validate GitHub loader configuration."""
        if 'github.com' not in url:
            raise ValueError(f"GitHub loader requires GitHub URL, got: {url}")
        
        # Check if we have access token
        params = config.params
        if not params.get('access_token') and not os.getenv('GITHUB_TOKEN'):
            raise ValueError("GitHub loader requires access token in config or GITHUB_TOKEN environment variable") 
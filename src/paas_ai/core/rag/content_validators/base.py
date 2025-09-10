"""
Base strategy interface for content validators.
"""

from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

from ...config import ContentValidatorConfig


class ContentValidatorStrategy(ABC):
    """Base strategy interface for content validators."""
    
    @abstractmethod
    def validate_documents(self, documents: List[Document], config: ContentValidatorConfig) -> List[Document]:
        """Validate documents and return filtered list."""
        pass
    
    @abstractmethod
    def is_valid_document(self, document: Document, config: ContentValidatorConfig) -> bool:
        """Check if a single document is valid."""
        pass
    
    @abstractmethod
    def validate_config(self, config: ContentValidatorConfig) -> None:
        """Validate configuration for this content validator strategy."""
        pass 
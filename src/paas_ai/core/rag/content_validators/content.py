"""
Content validator strategy implementation.
"""

from typing import List
from langchain_core.documents import Document

from .base import ContentValidatorStrategy
from ...config import ContentValidatorConfig


class ContentValidator(ContentValidatorStrategy):
    """Strategy for validating document content."""
    
    def validate_documents(self, documents: List[Document], config: ContentValidatorConfig) -> List[Document]:
        """Validate documents and return filtered list."""
        if not documents:
            return documents
        
        valid_documents = []
        for doc in documents:
            if self.is_valid_document(doc, config):
                valid_documents.append(doc)
        
        return valid_documents
    
    def is_valid_document(self, document: Document, config: ContentValidatorConfig) -> bool:
        """Check if a single document is valid."""
        content = document.page_content
        
        # Check empty content
        if config.skip_empty and not content.strip():
            return False
        
        # Check content length
        if len(content) < config.min_content_length:
            return False
        
        if len(content) > config.max_content_length:
            return False
        
        return True
    
    def validate_config(self, config: ContentValidatorConfig) -> None:
        """Validate configuration for this content validator strategy."""
        if config.min_content_length < 0:
            raise ValueError("min_content_length must be >= 0")
        
        if config.max_content_length <= 0:
            raise ValueError("max_content_length must be > 0")
        
        if config.min_content_length >= config.max_content_length:
            raise ValueError("min_content_length must be < max_content_length") 
"""
Base strategy interface for text splitters.
"""

from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter

from ..config import SplitterConfig


class SplitterStrategy(ABC):
    """Base strategy interface for text splitters."""
    
    @abstractmethod
    def create_splitter(self, config: SplitterConfig) -> TextSplitter:
        """Create a text splitter based on configuration."""
        pass
    
    @abstractmethod
    def validate_config(self, config: SplitterConfig) -> None:
        """Validate configuration for this splitter strategy."""
        pass
    
    def split_documents(self, config: SplitterConfig, documents: List[Document]) -> List[Document]:
        """Split documents using this strategy."""
        splitter = self.create_splitter(config)
        return splitter.split_documents(documents) 
"""
RAG (Retrieval-Augmented Generation) system for PaaS AI.

This module provides a complete RAG implementation following LangChain patterns,
including configurable document loaders, text splitters, embeddings, and vector stores.

Key Components:
- DocumentLoaderFactory: Creates appropriate loaders based on URL/file type
- TextSplitterFactory: Creates appropriate splitters based on content type
- EmbeddingsFactory: Creates embedding models based on configuration
- VectorStoreFactory: Creates vector stores with appropriate settings
- RetrieverFactory: Creates retrievers including ensemble options
- RAGProcessor: Main orchestrator for the entire RAG pipeline
- CitationEnricher: Adds citation metadata to documents during processing
- CitationFormatter: Formats citations according to different styles
"""

from .config import Config, ResourceConfig
from .pipeline import RAGProcessor, create_resource_from_url, ConfigurationError
from .loaders import DocumentLoaderFactory
from .splitters import TextSplitterFactory
from .embeddings import EmbeddingsFactory
from .vectorstore import VectorStoreFactory
from .retrievers import RetrieverFactory
from .citations import CitationEnricher, CitationFormatter, SourceReference, CitationAwareResult


__all__ = [
    'Config',
    'ResourceConfig', 
    'RAGProcessor',
    'create_resource_from_url',
    'ConfigurationError',
    'DocumentLoaderFactory',
    'TextSplitterFactory',
    'EmbeddingsFactory', 
    'VectorStoreFactory',
    'RetrieverFactory'
    'CitationEnricher',
    'CitationFormatter',
    'SourceReference', 
    'CitationAwareResult'
]

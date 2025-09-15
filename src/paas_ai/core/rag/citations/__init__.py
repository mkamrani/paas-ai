"""
Citation system for RAG.

Provides configurable citation strategies that integrate with the processing pipeline
to add precise source references and deep linking capabilities.
"""

from .models import SourceReference, CitationAwareResult
from .enricher import CitationEnricher
from .strategies import CitationStrategyRegistry
from .formatters import CitationFormatter

__all__ = [
    'SourceReference',
    'CitationAwareResult', 
    'CitationEnricher',
    'CitationStrategyRegistry',
    'CitationFormatter'
] 
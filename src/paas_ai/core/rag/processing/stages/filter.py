"""
Document filtering stage for processing pipeline.
"""

import re
from typing import List, Callable, Optional
from langchain_core.documents import Document

from ..base import ProcessingStage, ProcessingContext


class FilterStage(ProcessingStage):
    """Stage for filtering documents based on various criteria."""
    
    def __init__(self, 
                 name: str = "filter",
                 content_filters: Optional[List[Callable]] = None,
                 metadata_filters: Optional[List[Callable]] = None):
        super().__init__(name)
        self.content_filters = content_filters or []
        self.metadata_filters = metadata_filters or []
    
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """Filter documents."""
        documents = context.documents
        
        if not documents:
            # Set metadata even when no documents
            context.metadata.update({
                'filtered_count': 0,
                'removed_count': 0,
                'filter_rules_count': len(self.content_filters) + len(self.metadata_filters)
            })
            return context
        
        filtered_documents = []
        filtered_count = 0
        
        for doc in documents:
            if self._should_keep_document(doc):
                filtered_documents.append(doc)
            else:
                filtered_count += 1
        
        # Update context
        context.documents = filtered_documents
        context.metadata.update({
            'filtered_count': len(filtered_documents),
            'removed_count': filtered_count,
            'filter_rules_count': len(self.content_filters) + len(self.metadata_filters)
        })
        
        return context
    
    def _should_keep_document(self, doc: Document) -> bool:
        """Check if document should be kept."""
        # Apply content filters
        for content_filter in self.content_filters:
            if not content_filter(doc.page_content):
                return False
        
        # Apply metadata filters
        for metadata_filter in self.metadata_filters:
            if not metadata_filter(doc.metadata):
                return False
        
        return True


# Common filter functions
def no_duplicate_content_filter(seen_content: set):
    """Filter out documents with duplicate content."""
    def filter_func(content: str) -> bool:
        content_hash = hash(content)
        if content_hash in seen_content:
            return False
        seen_content.add(content_hash)
        return True
    return filter_func


def min_word_count_filter(min_words: int):
    """Filter out documents with too few words."""
    def filter_func(content: str) -> bool:
        return len(content.split()) >= min_words
    return filter_func


def language_filter(allowed_languages: List[str]):
    """Filter documents by language (requires language in metadata).""" 
    def filter_func(metadata: dict) -> bool:
        language = metadata.get('language', 'unknown')
        return language in allowed_languages
    return filter_func


def exclude_pattern_filter(patterns: List[str]):
    """Filter out documents containing specified patterns."""
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def filter_func(content: str) -> bool:
        for pattern in compiled_patterns:
            if pattern.search(content):
                return False
        return True
    return filter_func 
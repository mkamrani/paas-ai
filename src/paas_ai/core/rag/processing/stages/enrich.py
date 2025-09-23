"""
Document enrichment stage for processing pipeline.
"""

import time
from typing import List, Dict, Any
from langchain_core.documents import Document

from ..base import ProcessingStage, ProcessingContext


class EnrichStage(ProcessingStage):
    """Stage for enriching documents with metadata."""
    
    def __init__(self, name: str = "enrich", custom_enrichers: List = None):
        super().__init__(name)
        self.custom_enrichers = custom_enrichers or []
    
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """Enrich documents with metadata."""
        resource = context.resource
        documents = context.documents
        
        if not documents:
            # Set metadata even when no documents
            timestamp = time.time()
            context.metadata.update({
                'enriched_at': timestamp,
                'enricher_count': len(self.custom_enrichers)
            })
            return context
        
        # Standard metadata enrichment
        timestamp = time.time()
        
        for doc in documents:
            # Add resource metadata
            doc.metadata.update({
                'source_url': resource.url,
                'resource_type': resource.resource_type,
                'priority': resource.priority,
                'tags': resource.tags,
                'processed_at': timestamp,
                'pipeline_id': context.pipeline_id,
                **resource.metadata
            })
            
            # Add document-level metadata
            doc.metadata.update({
                'content_length': len(doc.page_content),
                'word_count': len(doc.page_content.split()),
                'char_count': len(doc.page_content)
            })
        
        # Apply citation enrichment if available and enabled
        citation_enricher = getattr(context, 'citation_enricher', None)
        if citation_enricher:
            documents = await citation_enricher(documents, context)
        
        # Apply custom enrichers
        for enricher in self.custom_enrichers:
            documents = await enricher(documents, context)
        
        # Update context
        context.documents = documents
        context.metadata.update({
            'enriched_at': timestamp,
            'enricher_count': len(self.custom_enrichers)
        })
        
        return context


class ContentEnricher:
    """Content-based document enricher."""
    
    async def __call__(self, documents: List[Document], context: ProcessingContext) -> List[Document]:
        """Enrich documents with content analysis."""
        for doc in documents:
            content = doc.page_content
            
            # Basic content analysis
            doc.metadata.update({
                'has_code': '```' in content or 'def ' in content or 'function ' in content,
                'has_urls': 'http' in content.lower(),
                'has_emails': '@' in content and '.' in content,
                'paragraph_count': content.count('\n\n') + 1,
                'sentence_count': content.count('.') + content.count('!') + content.count('?'),
            })
        
        return documents


class LanguageEnricher:
    """Language detection enricher."""
    
    def __init__(self):
        try:
            from langdetect import detect
            self.detect = detect
            self.available = True
        except ImportError:
            self.available = False
    
    async def __call__(self, documents: List[Document], context: ProcessingContext) -> List[Document]:
        """Detect language of documents."""
        if not self.available:
            return documents
        
        for doc in documents:
            try:
                language = self.detect(doc.page_content)
                doc.metadata['language'] = language
            except:
                doc.metadata['language'] = 'unknown'
        
        return documents 
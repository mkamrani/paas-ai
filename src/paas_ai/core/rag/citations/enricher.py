"""
Citation enricher for the RAG processing pipeline.

Integrates with the existing EnrichStage to add citation metadata to documents.
"""

import uuid
import hashlib
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from ..processing.base import ProcessingContext
from ...config.schemas import CitationConfig, ResourceType, CitationVerbosity
from .models import SourceReference, CitationMetadata
from .strategies import CitationStrategyRegistry


class CitationEnricher:
    """Enricher that adds citation metadata to documents during processing."""
    
    def __init__(self, citation_config: CitationConfig):
        self.config = citation_config
        self.strategy_registry = CitationStrategyRegistry()
    
    async def __call__(self, documents: List[Document], context: ProcessingContext) -> List[Document]:
        """Add citation metadata to documents."""
        
        if not self.config.enabled:
            return documents
        
        from paas_ai.utils.logging import get_logger
        logger = get_logger("paas_ai.rag.citations.enricher")
        
        logger.debug(f"Citation enricher called with {len(documents)} documents")
        logger.debug(f"Citation config enabled: {self.config.enabled}")
        logger.debug(f"Citation verbosity: {self.config.verbosity}")
        
        resource = context.resource
        resource_type = resource.resource_type
        
        # Get appropriate citation strategy
        strategy_name = self.config.strategies.get(resource_type, "default_citation")
        strategy = self.strategy_registry.get_strategy(strategy_name)
        
        # Get effective verbosity for this resource type
        effective_verbosity = self.config.resource_overrides.get(
            resource_type, 
            self.config.verbosity
        )
        
        for doc in documents:
            try:
                logger.debug(f"Processing document chunk: {doc.page_content[:100]}...")
                
                # Generate unique chunk ID
                chunk_id = self._generate_chunk_id(doc, resource)
                logger.debug(f"Generated chunk_id: {chunk_id}")
                
                # Extract citation metadata using strategy
                citation_metadata = strategy.extract_citation_metadata(doc, context)
                logger.debug(f"Extracted citation metadata")
                
                # Create source reference
                source_ref = self._create_source_reference(
                    doc, resource, chunk_id, citation_metadata, context
                )
                logger.debug(f"Created source reference")
                
                # Add citation information to document metadata
                doc.metadata.update({
                    'citation_reference': source_ref.model_dump(),
                    'citation_metadata': citation_metadata.model_dump(),
                    'citation_enabled': True,
                    'citation_verbosity': effective_verbosity.value if hasattr(effective_verbosity, 'value') else effective_verbosity,
                    'citation_strategy': strategy_name
                })
                logger.debug(f"Added citation metadata to document")
                
            except Exception as e:
                logger.error(f"ERROR in citation enricher: {e}")
                logger.debug(f"Citation enricher exception details", exc_info=True)
        
        return documents
    
    def _generate_chunk_id(self, doc: Document, resource) -> str:
        """Generate a unique identifier for this document chunk."""
        # Create deterministic ID based on content and source
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
        source_hash = hashlib.md5(resource.url.encode()).hexdigest()[:8]
        return f"{source_hash}_{content_hash}_{str(uuid.uuid4())[:8]}"
    
    def _create_source_reference(
        self, 
        doc: Document, 
        resource, 
        chunk_id: str,
        citation_metadata: CitationMetadata,
        context: ProcessingContext
    ) -> SourceReference:
        """Create a comprehensive source reference for the document."""
        
        # Extract basic location information
        page_number = doc.metadata.get('page')
        section_title = self._extract_section_title(doc, citation_metadata)
        
        # Extract quote if enabled
        exact_quote = None
        if self.config.include_quotes:
            exact_quote = self._extract_relevant_quote(doc.page_content)
        
        # Create source reference
        source_ref = SourceReference(
            source_url=resource.url,
            resource_type=resource.resource_type,
            document_id=str(context.pipeline_id),
            chunk_id=chunk_id,
            page_number=page_number,
            section_title=section_title,
            exact_quote=exact_quote,
            confidence_score=self._calculate_confidence_score(doc, citation_metadata),
            metadata={
                'tags': resource.tags,
                'priority': resource.priority,
                'processed_at': doc.metadata.get('processed_at'),
                'loader_type': doc.metadata.get('source_loader'),
                **citation_metadata.content_type_markers
            }
        )
        
        return source_ref
    
    def _extract_section_title(self, doc: Document, citation_metadata: CitationMetadata) -> Optional[str]:
        """Extract the most relevant section title for this document chunk."""
        
        # Look for headings in citation metadata
        if citation_metadata.headings:
            # Find the most appropriate heading based on position
            return citation_metadata.headings[0].get('text', None)
        
        # Fallback to document metadata
        return doc.metadata.get('section', doc.metadata.get('title'))
    
    def _extract_relevant_quote(self, content: str) -> Optional[str]:
        """Extract a relevant quote from the document content."""
        
        if not content.strip():
            return None
        
        # Take first sentence or up to max_quote_length
        sentences = content.split('. ')
        if sentences:
            quote = sentences[0]
            if len(quote) > self.config.max_quote_length:
                quote = quote[:self.config.max_quote_length].rsplit(' ', 1)[0] + "..."
            return quote
        
        # Fallback to truncated content
        if len(content) > self.config.max_quote_length:
            return content[:self.config.max_quote_length].rsplit(' ', 1)[0] + "..."
        
        return content
    
    def _calculate_confidence_score(self, doc: Document, citation_metadata: CitationMetadata) -> float:
        """Calculate a confidence score for this citation."""
        
        if not self.config.include_confidence:
            return 1.0
        
        score = 1.0
        
        # Reduce confidence if missing key metadata
        if not doc.metadata.get('page') and doc.metadata.get('source_loader') == 'pdf':
            score -= 0.1
        
        if not citation_metadata.headings:
            score -= 0.1
            
        # Boost confidence for structured content
        if citation_metadata.content_type_markers.get('has_structure', False):
            score += 0.1
        
        return max(0.0, min(1.0, score)) 
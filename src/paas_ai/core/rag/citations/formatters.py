"""
Citation formatters for different output styles.
"""

from typing import List, Dict, Any
from ...config.schemas import CitationFormat, CitationVerbosity
from .models import SourceReference
from .strategies import CitationStrategyRegistry


class CitationFormatter:
    """Formats citations according to specified styles."""
    
    def __init__(self):
        self.strategy_registry = CitationStrategyRegistry()
    
    def format_citation(
        self, 
        source_ref: SourceReference, 
        verbosity: CitationVerbosity,
        format_style: CitationFormat,
        strategy_name: str = "default_citation"
    ) -> str:
        """Format a single citation."""
        
        if verbosity == CitationVerbosity.NONE:
            return ""
        
        strategy = self.strategy_registry.get_strategy(strategy_name)
        base_citation = strategy.format_citation(source_ref, verbosity)
        
        if format_style == CitationFormat.INLINE:
            return base_citation
        elif format_style == CitationFormat.FOOTNOTE:
            return self._format_footnote(base_citation, source_ref)
        elif format_style == CitationFormat.ACADEMIC:
            return self._format_academic(source_ref, verbosity)
        elif format_style == CitationFormat.STRUCTURED:
            return self._format_structured(source_ref, verbosity)
        else:
            return base_citation
    
    def format_multiple_citations(
        self,
        source_refs: List[SourceReference],
        verbosity: CitationVerbosity,
        format_style: CitationFormat,
        strategy_names: List[str] = None
    ) -> List[str]:
        """Format multiple citations."""
        
        if not strategy_names:
            strategy_names = ["default_citation"] * len(source_refs)
        
        citations = []
        for i, source_ref in enumerate(source_refs):
            strategy_name = strategy_names[i] if i < len(strategy_names) else "default_citation"
            citation = self.format_citation(source_ref, verbosity, format_style, strategy_name)
            if citation:  # Only add non-empty citations
                citations.append(citation)
        
        return citations
    
    def _format_footnote(self, base_citation: str, source_ref: SourceReference) -> str:
        """Format as footnote reference."""
        # Strip brackets and create footnote format
        clean_citation = base_citation.strip('[]')
        return f"[^{hash(source_ref.chunk_id) % 1000}]"
    
    def _format_academic(self, source_ref: SourceReference, verbosity: CitationVerbosity) -> str:
        """Format in academic citation style."""
        
        parts = []
        
        # Extract source name
        if source_ref.source_url.endswith('.pdf'):
            source_name = source_ref.source_url.split('/')[-1].replace('.pdf', '')
        else:
            source_name = self._extract_domain(source_ref.source_url)
        
        parts.append(source_name)
        
        # Add location information
        if source_ref.page_number:
            parts.append(f"{source_ref.page_number}")
        elif source_ref.section_title:
            parts.append(f'"{source_ref.section_title}"')
        
        # Add access date for web sources
        if source_ref.source_url.startswith('http'):
            if source_ref.metadata.get('processed_at'):
                parts.append(f"accessed {source_ref.metadata['processed_at']}")
        
        # Add URL
        parts.append(source_ref.source_url)
        
        return f"({', '.join(parts)})"
    
    def _format_structured(self, source_ref: SourceReference, verbosity: CitationVerbosity) -> str:
        """Format as structured data (JSON-like)."""
        
        citation_data = {
            "source": self._extract_domain(source_ref.source_url),
            "url": source_ref.source_url,
            "type": source_ref.resource_type
        }
        
        if source_ref.page_number:
            citation_data["page"] = source_ref.page_number
        
        if source_ref.section_title:
            citation_data["section"] = source_ref.section_title
        
        if verbosity in [CitationVerbosity.DETAILED, CitationVerbosity.FORENSIC]:
            if source_ref.exact_quote:
                citation_data["quote"] = source_ref.exact_quote
            
            if source_ref.confidence_score != 1.0:
                citation_data["confidence"] = source_ref.confidence_score
        
        # Format as readable string
        parts = [f"{k}: {v}" for k, v in citation_data.items()]
        return f"{{{', '.join(parts)}}}"
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        if url.startswith(('http://', 'https://')):
            parts = url.split('/')
            if len(parts) > 2:
                domain = parts[2]
                if domain.startswith('www.'):
                    domain = domain[4:]
                return domain
            return url
        else:
            return url.split('/')[-1] 
"""
Citation strategies for different resource types.

Provides specialized citation extraction and formatting based on content type.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_core.documents import Document

from ..processing.base import ProcessingContext
from paas_ai.core.config.schemas import ResourceType, CitationVerbosity
from .models import CitationMetadata, SourceReference


class CitationStrategy(ABC):
    """Base strategy for citation generation and formatting."""
    
    @abstractmethod
    def extract_citation_metadata(self, document: Document, context: ProcessingContext) -> CitationMetadata:
        """Extract citation-relevant metadata from document."""
        pass
    
    @abstractmethod
    def format_citation(self, source_ref: SourceReference, verbosity: CitationVerbosity) -> str:
        """Format citation according to strategy and verbosity level."""
        pass
    
    @abstractmethod
    def generate_citation_link(self, source_ref: SourceReference) -> Optional[str]:
        """Generate deep link to source if possible."""
        pass


class DefaultCitationStrategy(CitationStrategy):
    """Default citation strategy for general content."""
    
    def extract_citation_metadata(self, document: Document, context: ProcessingContext) -> CitationMetadata:
        """Extract basic citation metadata."""
        
        metadata = CitationMetadata()
        
        # Basic structure analysis
        content = document.page_content
        metadata.content_type_markers = {
            'has_structure': bool(document.metadata.get('section') or document.metadata.get('title')),
            'has_page_number': bool(document.metadata.get('page')),
            'content_length': len(content),
            'word_count': len(content.split())
        }
        
        return metadata
    
    def format_citation(self, source_ref: SourceReference, verbosity: CitationVerbosity) -> str:
        """Format citation based on verbosity level."""
        
        if verbosity == CitationVerbosity.NONE:
            return ""
        elif verbosity == CitationVerbosity.MINIMAL:
            return f"[{self._get_source_name(source_ref.source_url)}]"
        elif verbosity == CitationVerbosity.STANDARD:
            parts = [self._get_source_name(source_ref.source_url)]
            if source_ref.page_number:
                parts.append(f"p. {source_ref.page_number}")
            elif source_ref.section_title:
                parts.append(source_ref.section_title)
            return f"[{', '.join(parts)}]"
        elif verbosity == CitationVerbosity.DETAILED:
            parts = [self._get_source_name(source_ref.source_url)]
            if source_ref.page_number:
                parts.append(f"Page {source_ref.page_number}")
            if source_ref.section_title:
                parts.append(f"Section '{source_ref.section_title}'")
            return f"[{', '.join(parts)}]"
        else:  # FORENSIC
            parts = [self._get_source_name(source_ref.source_url)]
            if source_ref.page_number:
                parts.append(f"Page {source_ref.page_number}")
            if source_ref.section_title:
                parts.append(f"Section '{source_ref.section_title}'")
            citation = f"[{', '.join(parts)}]"
            if source_ref.exact_quote:
                citation += f': "{source_ref.exact_quote}"'
            return citation
    
    def generate_citation_link(self, source_ref: SourceReference) -> Optional[str]:
        """Generate link to source."""
        return source_ref.source_url
    
    def _get_source_name(self, url: str) -> str:
        """Extract a readable source name from URL."""
        if url.startswith(('http://', 'https://')):
            # Extract domain or last path component
            parts = url.split('/')
            if len(parts) > 2:
                domain = parts[2]
                if domain.startswith('www.'):
                    domain = domain[4:]
                return domain
            return url
        else:
            # Local file - extract filename
            return url.split('/')[-1]


class TechnicalCitationStrategy(CitationStrategy):
    """Citation strategy for technical documentation (DSL resources)."""
    
    def extract_citation_metadata(self, document: Document, context: ProcessingContext) -> CitationMetadata:
        """Extract technical documentation metadata."""
        
        metadata = CitationMetadata()
        content = document.page_content
        
        # Analyze for technical content
        metadata.content_type_markers = {
            'has_code': '```' in content or 'def ' in content or 'function ' in content,
            'has_api_refs': any(word in content.lower() for word in ['api', 'endpoint', 'method', 'parameter']),
            'has_examples': 'example' in content.lower() or 'sample' in content.lower(),
            'has_version_info': any(word in content for word in ['v1', 'v2', 'version', 'release']),
            'content_length': len(content),
            'has_structure': bool(document.metadata.get('section'))
        }
        
        # Extract headings that look technical
        if document.metadata.get('section'):
            metadata.headings = [{'text': document.metadata['section'], 'level': 1}]
        
        return metadata
    
    def format_citation(self, source_ref: SourceReference, verbosity: CitationVerbosity) -> str:
        """Format technical citation."""
        
        if verbosity == CitationVerbosity.NONE:
            return ""
        elif verbosity == CitationVerbosity.MINIMAL:
            return f"[{self._get_api_name(source_ref)}]"
        elif verbosity == CitationVerbosity.STANDARD:
            parts = [self._get_api_name(source_ref)]
            if source_ref.section_title:
                parts.append(source_ref.section_title)
            return f"[{', '.join(parts)}]"
        elif verbosity == CitationVerbosity.DETAILED:
            parts = [self._get_api_name(source_ref)]
            if source_ref.metadata.get('has_version_info'):
                parts.append("API Documentation")
            if source_ref.section_title:
                parts.append(f"Section: {source_ref.section_title}")
            return f"[{', '.join(parts)}]"
        else:  # FORENSIC
            parts = [self._get_api_name(source_ref)]
            if source_ref.section_title:
                parts.append(f"Section: {source_ref.section_title}")
            citation = f"[{', '.join(parts)}]"
            if source_ref.exact_quote:
                citation += f': "{source_ref.exact_quote}"'
            return citation
    
    def generate_citation_link(self, source_ref: SourceReference) -> Optional[str]:
        """Generate deep link for technical docs."""
        base_url = source_ref.source_url
        if source_ref.section_title and base_url.startswith('http'):
            # Create anchor link
            anchor = source_ref.section_title.lower().replace(' ', '-').replace('(', '').replace(')', '')
            return f"{base_url}#{anchor}"
        return base_url
    
    def _get_api_name(self, source_ref: SourceReference) -> str:
        """Extract API or service name."""
        url = source_ref.source_url
        if 'kubernetes.io' in url:
            return "Kubernetes Docs"
        elif 'docker.com' in url:
            return "Docker Docs"
        elif source_ref.metadata.get('tags'):
            # Use first tag as API name
            return source_ref.metadata['tags'][0].title()
        else:
            return self._get_source_name(url)
    
    def _get_source_name(self, url: str) -> str:
        """Extract source name from URL."""
        if url.startswith(('http://', 'https://')):
            parts = url.split('/')
            if len(parts) > 2:
                domain = parts[2]
                if domain.startswith('www.'):
                    domain = domain[4:]
                return domain.split('.')[0].title() + " Docs"
            return url
        else:
            return url.split('/')[-1]


class WebCitationStrategy(DefaultCitationStrategy):
    """Citation strategy for web content."""
    
    def generate_citation_link(self, source_ref: SourceReference) -> Optional[str]:
        """Generate anchor links for web content."""
        if source_ref.section_title and source_ref.source_url.startswith('http'):
            anchor = self._create_anchor(source_ref.section_title)
            return f"{source_ref.source_url}#{anchor}"
        return source_ref.source_url
    
    def _create_anchor(self, section_title: str) -> str:
        """Create URL anchor from section title."""
        return section_title.lower().replace(' ', '-').replace('(', '').replace(')', '').replace('.', '')


class PolicyCitationStrategy(CitationStrategy):
    """Citation strategy for policy and guidelines documents."""
    
    def extract_citation_metadata(self, document: Document, context: ProcessingContext) -> CitationMetadata:
        """Extract policy document metadata."""
        
        metadata = CitationMetadata()
        content = document.page_content
        
        # Look for policy-specific markers
        metadata.content_type_markers = {
            'has_requirements': any(word in content.lower() for word in ['must', 'shall', 'required', 'mandatory']),
            'has_sections': bool(document.metadata.get('section')),
            'has_numbering': any(word in content for word in ['1.', '2.', 'a)', 'b)']),
            'has_policy_language': any(word in content.lower() for word in ['policy', 'guideline', 'compliance', 'standard']),
            'content_length': len(content)
        }
        
        return metadata
    
    def format_citation(self, source_ref: SourceReference, verbosity: CitationVerbosity) -> str:
        """Format policy citation."""
        
        if verbosity == CitationVerbosity.NONE:
            return ""
        elif verbosity == CitationVerbosity.MINIMAL:
            return f"[{self._get_policy_name(source_ref)}]"
        elif verbosity == CitationVerbosity.STANDARD:
            parts = [self._get_policy_name(source_ref)]
            if source_ref.section_title:
                parts.append(f"Section {source_ref.section_title}")
            return f"[{', '.join(parts)}]"
        elif verbosity == CitationVerbosity.DETAILED:
            parts = [self._get_policy_name(source_ref)]
            if source_ref.section_title:
                parts.append(f"Section {source_ref.section_title}")
            if source_ref.page_number:
                parts.append(f"Page {source_ref.page_number}")
            return f"[{', '.join(parts)}]"
        else:  # FORENSIC
            parts = [self._get_policy_name(source_ref)]
            if source_ref.section_title:
                parts.append(f"Section {source_ref.section_title}")
            if source_ref.page_number:
                parts.append(f"Page {source_ref.page_number}")
            citation = f"[{', '.join(parts)}]"
            if source_ref.exact_quote:
                citation += f': "{source_ref.exact_quote}"'
            return citation
    
    def generate_citation_link(self, source_ref: SourceReference) -> Optional[str]:
        """Generate link for policy documents."""
        return source_ref.source_url
    
    def _get_policy_name(self, source_ref: SourceReference) -> str:
        """Extract policy name."""
        url = source_ref.source_url
        if url.endswith('.pdf'):
            filename = url.split('/')[-1].replace('.pdf', '')
            return filename.replace('_', ' ').title()
        else:
            return self._get_source_name(url)
    
    def _get_source_name(self, url: str) -> str:
        """Extract source name."""
        if url.startswith(('http://', 'https://')):
            parts = url.split('/')
            if len(parts) > 2:
                return parts[2].replace('www.', '').split('.')[0].title() + " Policy"
            return url
        else:
            return url.split('/')[-1]


class CitationStrategyRegistry:
    """Registry for citation strategies."""
    
    def __init__(self):
        self._strategies = {
            'default_citation': DefaultCitationStrategy(),
            'technical_citation': TechnicalCitationStrategy(),
            'web_citation': WebCitationStrategy(),
            'policy_citation': PolicyCitationStrategy(),
            'rule_citation': PolicyCitationStrategy(),  # Reuse policy strategy for rules
        }
    
    def get_strategy(self, strategy_name: str) -> CitationStrategy:
        """Get strategy by name."""
        return self._strategies.get(strategy_name, self._strategies['default_citation'])
    
    def register_strategy(self, name: str, strategy: CitationStrategy):
        """Register a new strategy."""
        self._strategies[name] = strategy
    
    def list_strategies(self) -> list:
        """List available strategies."""
        return list(self._strategies.keys()) 
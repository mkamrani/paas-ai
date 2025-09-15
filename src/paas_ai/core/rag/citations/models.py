"""
Core citation data models.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from ...config.schemas import ResourceType, CitationVerbosity


class SourceReference(BaseModel):
    """Comprehensive source reference with hierarchical context."""
    
    # Document identifiers
    source_url: str
    resource_type: ResourceType
    document_id: str
    
    # Hierarchical location tracking
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    paragraph_number: Optional[int] = None
    line_number: Optional[int] = None
    
    # Content tracking
    exact_quote: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    surrounding_context: Optional[str] = None
    
    # Chunk reference
    chunk_id: str
    confidence_score: float = 1.0
    
    # Resource-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class CitationAwareResult(BaseModel):
    """Enhanced search result with citation information."""
    
    content: str
    score: float
    source_reference: SourceReference
    formatted_citation: str
    citation_link: Optional[str] = None
    
    # Additional citation context
    supporting_quotes: List[str] = Field(default_factory=list)
    confidence_indicators: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class CitationMetadata(BaseModel):
    """Metadata extracted during document processing for citation purposes."""
    
    # Structure information
    document_structure: Dict[str, Any] = Field(default_factory=dict)
    headings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Location markers
    page_breaks: List[int] = Field(default_factory=list)
    section_markers: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Content analysis
    content_type_markers: Dict[str, Any] = Field(default_factory=dict)
    
    # Deep linking info
    anchor_points: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True 
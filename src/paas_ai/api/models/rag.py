"""
Pydantic models for RAG-related API endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum
from .base import SuccessResponse, PaginatedResponse


class ResourceType(str, Enum):
    """Resource type enumeration."""
    DSL = "dsl"
    DOCUMENTATION = "documentation"
    API = "api"
    CODE = "code"
    OTHER = "other"


class AddResourceRequest(BaseModel):
    """Request model for adding a single resource."""
    url: str = Field(
        ...,
        description="URL or file path of the resource",
        example="https://example.com/docs/api.md"
    )
    resource_type: ResourceType = Field(
        default=ResourceType.DSL,
        description="Type of the resource"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing the resource",
        example=["api", "documentation", "v1"]
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority level (1-10, higher is more important)"
    )
    config_profile: Optional[str] = Field(
        default=None,
        description="Override the default config profile"
    )


class BatchResourceRequest(BaseModel):
    """Request model for adding multiple resources."""
    resources: List[AddResourceRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of resources to add"
    )
    config_profile: Optional[str] = Field(
        default=None,
        description="Override the default config profile"
    )


class ResourceProcessingResult(BaseModel):
    """Result of processing a single resource."""
    url: str = Field(description="Resource URL")
    success: bool = Field(description="Whether processing was successful")
    documents_count: int = Field(description="Number of documents extracted")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    processing_time: float = Field(description="Processing time in seconds")


class AddResourceResponse(BaseModel):
    """Response for adding resources."""
    total_resources: int = Field(description="Total number of resources processed")
    successful: int = Field(description="Number of successfully processed resources")
    failed: int = Field(description="Number of failed resources")
    total_documents: int = Field(description="Total documents extracted")
    results: List[ResourceProcessingResult] = Field(description="Detailed results per resource")
    processing_time: float = Field(description="Total processing time in seconds")


class AddResourceCompleteResponse(SuccessResponse):
    """Complete response for add resource endpoint."""
    data: AddResourceResponse


class SearchRequest(BaseModel):
    """Request model for searching the knowledge base."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query",
        example="authentication flow"
    )
    resource_type: Optional[ResourceType] = Field(
        default=None,
        description="Filter by resource type"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Filter by tags"
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of results to return"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in results"
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold"
    )


class SearchResult(BaseModel):
    """Individual search result."""
    content: str = Field(description="Document content")
    score: float = Field(description="Similarity score")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Document metadata including source, tags, etc."
    )


class SearchResponse(BaseModel):
    """Response for search endpoint."""
    query: str = Field(description="Original search query")
    results: List[SearchResult] = Field(description="Search results")
    total_results: int = Field(description="Total number of results found")
    processing_time: float = Field(description="Search processing time in seconds")


class SearchCompleteResponse(SuccessResponse):
    """Complete response for search endpoint."""
    data: SearchResponse


class RAGStatusResponse(BaseModel):
    """RAG system status information."""
    vectorstore_status: str = Field(description="Vector store status")
    total_documents: int = Field(description="Total documents in the knowledge base")
    collections: List[str] = Field(description="Available collections")
    embedding_model: str = Field(description="Current embedding model")
    last_updated: Optional[str] = Field(description="Last update timestamp")
    configuration: Dict[str, Any] = Field(description="Current RAG configuration")


class RAGStatusCompleteResponse(SuccessResponse):
    """Complete response for RAG status endpoint."""
    data: RAGStatusResponse


class ResourceInfo(BaseModel):
    """Information about a stored resource."""
    url: str = Field(description="Resource URL")
    resource_type: ResourceType = Field(description="Resource type")
    tags: List[str] = Field(description="Resource tags")
    priority: int = Field(description="Resource priority")
    documents_count: int = Field(description="Number of documents from this resource")
    added_at: str = Field(description="When the resource was added")
    last_updated: str = Field(description="Last update time")


class ListResourcesResponse(BaseModel):
    """Response for listing resources."""
    resources: List[ResourceInfo] = Field(description="List of resources")
    total_count: int = Field(description="Total number of resources")


class ListResourcesCompleteResponse(PaginatedResponse):
    """Complete paginated response for listing resources."""
    data: ListResourcesResponse 
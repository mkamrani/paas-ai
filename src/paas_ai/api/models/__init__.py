"""
API models for PaaS AI.
"""

from .agent import (
    AgentConfigSummary,
    AgentInfoResponse,
    AgentToolInfo,
    ChatCompleteResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
)
from .base import (
    BaseResponse,
    ErrorResponse,
    HealthStatus,
    PaginatedResponse,
    PaginationParams,
    ResponseStatus,
    SuccessResponse,
)
from .rag import (
    AddResourceCompleteResponse,
    AddResourceRequest,
    AddResourceResponse,
    BatchResourceRequest,
    ListResourcesCompleteResponse,
    ListResourcesResponse,
    RAGStatusCompleteResponse,
    RAGStatusResponse,
    ResourceInfo,
    ResourceProcessingResult,
    ResourceType,
    SearchCompleteResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

__all__ = [
    # Base models
    "ResponseStatus",
    "BaseResponse",
    "SuccessResponse",
    "ErrorResponse",
    "PaginationParams",
    "PaginatedResponse",
    "HealthStatus",
    # Agent models
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatCompleteResponse",
    "AgentToolInfo",
    "AgentConfigSummary",
    "AgentInfoResponse",
    # RAG models
    "ResourceType",
    "AddResourceRequest",
    "BatchResourceRequest",
    "ResourceProcessingResult",
    "AddResourceResponse",
    "AddResourceCompleteResponse",
    "SearchRequest",
    "SearchResult",
    "SearchResponse",
    "SearchCompleteResponse",
    "RAGStatusResponse",
    "RAGStatusCompleteResponse",
    "ResourceInfo",
    "ListResourcesResponse",
    "ListResourcesCompleteResponse",
]

"""
API models for PaaS AI.
"""

from .base import (
    ResponseStatus,
    BaseResponse,
    SuccessResponse,
    ErrorResponse,
    PaginationParams,
    PaginatedResponse,
    HealthStatus
)

from .agent import (
    AgentQuestionRequest,
    AgentResponse,
    AgentQuestionResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatCompleteResponse,
    AgentToolInfo,
    AgentConfigSummary,
    AgentInfoResponse
)

from .rag import (
    ResourceType,
    AddResourceRequest,
    BatchResourceRequest,
    ResourceProcessingResult,
    AddResourceResponse,
    AddResourceCompleteResponse,
    SearchRequest,
    SearchResult,
    SearchResponse,
    SearchCompleteResponse,
    RAGStatusResponse,
    RAGStatusCompleteResponse,
    ResourceInfo,
    ListResourcesResponse,
    ListResourcesCompleteResponse
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
    "AgentQuestionRequest",
    "AgentResponse",
    "AgentQuestionResponse", 
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
    "ListResourcesCompleteResponse"
]

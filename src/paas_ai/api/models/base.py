"""
Base Pydantic models for API responses and common structures.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ResponseStatus(str, Enum):
    """Response status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class BaseResponse(BaseModel):
    """Base response model for all API endpoints."""
    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})
    
    status: ResponseStatus
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class SuccessResponse(BaseResponse):
    """Success response with data."""
    status: ResponseStatus = ResponseStatus.SUCCESS
    data: Any


class ErrorResponse(BaseResponse):
    """Error response model."""
    status: ResponseStatus = ResponseStatus.ERROR
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=10, ge=1, le=100, description="Page size")


class PaginatedResponse(SuccessResponse):
    """Paginated response model."""
    pagination: Dict[str, Any] = Field(
        description="Pagination metadata",
        example={
            "page": 1,
            "size": 10, 
            "total": 100,
            "pages": 10,
            "has_next": True,
            "has_prev": False
        }
    )


class HealthStatus(BaseModel):
    """Health check response."""
    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})
    
    status: str = "healthy"
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, str] = Field(
        description="Status of various system components",
        example={
            "database": "healthy",
            "embeddings": "healthy", 
            "vectorstore": "healthy",
            "llm": "healthy"
        }
    ) 
"""
Pydantic models for agent-related API endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import SuccessResponse


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str = Field(
        ..., pattern="^(human|assistant|system)$", description="Role of the message sender"
    )
    content: str = Field(..., min_length=1, description="Message content")
    timestamp: Optional[str] = Field(default=None, description="Message timestamp")


class ChatRequest(BaseModel):
    """Request model for chat conversation."""

    messages: List[ChatMessage] = Field(..., min_items=1, description="Conversation history")
    config_profile: Optional[str] = Field(
        default=None, description="Override the default config profile"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    message: ChatMessage = Field(description="Agent's response message")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")
    processing_time: float = Field(description="Processing time in seconds")


class ChatCompleteResponse(SuccessResponse):
    """Complete response for chat endpoint."""

    data: ChatResponse


class AgentToolInfo(BaseModel):
    """Information about available agent tools."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Tool parameter schema")


class AgentConfigSummary(BaseModel):
    """Agent configuration summary."""

    llm: Dict[str, Any] = Field(description="LLM configuration")
    embedding: Dict[str, Any] = Field(description="Embedding configuration")
    vectorstore: Dict[str, Any] = Field(description="Vector store configuration")
    tools: List[AgentToolInfo] = Field(description="Available tools")


class AgentInfoResponse(SuccessResponse):
    """Response for agent info endpoint."""

    data: AgentConfigSummary

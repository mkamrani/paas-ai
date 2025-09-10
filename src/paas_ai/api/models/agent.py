"""
Pydantic models for agent-related API endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .base import SuccessResponse


class AgentQuestionRequest(BaseModel):
    """Request model for asking the agent a question."""
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="The question to ask the agent",
        example="What features does the test document mention?"
    )
    config_profile: Optional[str] = Field(
        default=None,
        description="Override the default config profile",
        example="local"
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source information in the response"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=4000,
        description="Maximum tokens in the response"
    )


class AgentResponse(BaseModel):
    """Response from the agent."""
    answer: str = Field(description="The agent's answer")
    sources: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Sources used to generate the answer"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the answer"
    )
    processing_time: float = Field(description="Time taken to process the question in seconds")


class AgentQuestionResponse(SuccessResponse):
    """Complete response for agent question endpoint."""
    data: AgentResponse


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(
        ..., 
        pattern="^(human|assistant|system)$",
        description="Role of the message sender"
    )
    content: str = Field(..., min_length=1, description="Message content")
    timestamp: Optional[str] = Field(default=None, description="Message timestamp")


class ChatRequest(BaseModel):
    """Request model for chat conversation."""
    messages: List[ChatMessage] = Field(
        ...,
        min_items=1,
        description="Conversation history"
    )
    config_profile: Optional[str] = Field(
        default=None,
        description="Override the default config profile"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    message: ChatMessage = Field(description="Agent's response message")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation identifier"
    )
    processing_time: float = Field(description="Processing time in seconds")


class ChatCompleteResponse(SuccessResponse):
    """Complete response for chat endpoint."""
    data: ChatResponse


class AgentToolInfo(BaseModel):
    """Information about available agent tools."""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tool parameter schema"
    )


class AgentConfigSummary(BaseModel):
    """Agent configuration summary."""
    llm: Dict[str, Any] = Field(description="LLM configuration")
    embedding: Dict[str, Any] = Field(description="Embedding configuration")
    vectorstore: Dict[str, Any] = Field(description="Vector store configuration")
    tools: List[AgentToolInfo] = Field(description="Available tools")


class AgentInfoResponse(SuccessResponse):
    """Response for agent info endpoint."""
    data: AgentConfigSummary 
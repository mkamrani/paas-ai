"""
Agent router for RAG agent API endpoints.
"""

import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from paas_ai.core.agents.multi_agent_system import MultiAgentSystem
from paas_ai.core.config import Config, load_config
from paas_ai.utils.logging import get_logger

from ..middleware.security import APIKeyAuth
from ..models import (
    AgentConfigSummary,
    AgentInfoResponse,
    AgentToolInfo,
    ChatCompleteResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)

logger = get_logger("paas_ai.api.agent")
router = APIRouter(prefix="/agent", tags=["agent"])

# Security dependency
security = APIKeyAuth()


def get_config(config_profile: Optional[str] = None) -> Config:
    """Get configuration with optional profile override."""
    # Default to 'local' profile for API to match data ingestion
    profile = config_profile or "local"

    logger.info(f"Using config profile: {profile}")

    # Load configuration with the specified profile
    # Since we need to match the profile used for data ingestion, use 'local'
    # TODO: Implement proper profile loading from config system
    config = load_config()

    # For now, ensure we use local/SentenceTransformers profile settings
    if profile == "local":
        # Override to use local settings that match data ingestion
        from ...core.config.schemas import DEFAULT_CONFIG_PROFILES

        local_profile = DEFAULT_CONFIG_PROFILES["local"]

        # Update embedding config to match ingestion
        config.embedding = local_profile.embedding
        config.vectorstore = local_profile.vectorstore
        config.llm = local_profile.llm

        logger.info(f"✅ Using local profile with {config.embedding.type} embeddings")

    return config


@router.post(
    "/chat",
    response_model=ChatCompleteResponse,
    summary="Chat with the agent",
    description="Have a conversation with the RAG agent using message history.",
)
async def chat_with_agent(
    request: ChatRequest,
    api_key: str = Depends(security),
    x_config_profile: Optional[str] = Header(None, alias="X-Config-Profile"),
) -> ChatCompleteResponse:
    """Chat with the agent using conversation history."""
    start_time = time.time()

    try:
        # Use config profile from request or header
        config_profile = request.config_profile or x_config_profile

        # Load configuration
        config = get_config(config_profile)
        logger.info(f"Processing chat with {len(request.messages)} messages")

        # Initialize agent
        agent = MultiAgentSystem(config)

        # Convert request messages to LangChain format
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in request.messages:
            if msg.role == "human":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))

        # Get agent response
        response_content = agent.chat(lc_messages)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create response message
        response_message = ChatMessage(role="assistant", content=response_content)

        # Prepare response
        response_data = ChatResponse(
            message=response_message,
            conversation_id=None,  # TODO: Implement conversation tracking
            processing_time=processing_time,
        )

        logger.info(f"Chat processed successfully in {processing_time:.2f}s")

        return ChatCompleteResponse(message="Chat processed successfully", data=response_data)

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Failed to process chat: {str(e)}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat: {str(e)}",
        )


@router.get(
    "/info",
    response_model=AgentInfoResponse,
    summary="Get agent information",
    description="Retrieve information about the agent configuration and available tools.",
)
async def get_agent_info(
    api_key: str = Depends(security),
    config_profile: Optional[str] = None,
    x_config_profile: Optional[str] = Header(None, alias="X-Config-Profile"),
) -> AgentInfoResponse:
    """Get agent configuration and tool information."""
    try:
        # Use config profile from query param or header
        profile = config_profile or x_config_profile

        # Load configuration
        config = get_config(profile)

        # Initialize agent to get tool info
        agent = MultiAgentSystem(config)

        # Get configuration summary
        config_summary = agent.get_config_summary()

        # Get tool information
        tools_info = []
        for tool_info in agent.get_available_tools():
            tools_info.append(
                AgentToolInfo(
                    name=tool_info["name"],
                    description=tool_info["description"],
                    parameters=tool_info.get("args_schema"),
                )
            )

        # Prepare response
        response_data = AgentConfigSummary(
            llm=config_summary["llm"],
            embedding=config_summary["embedding"],
            vectorstore=config_summary["vectorstore"],
            tools=tools_info,
        )

        return AgentInfoResponse(
            message="Agent information retrieved successfully", data=response_data
        )

    except Exception as e:
        logger.error(f"Failed to get agent info: {str(e)}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent information: {str(e)}",
        )

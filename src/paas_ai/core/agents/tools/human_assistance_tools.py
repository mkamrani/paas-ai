"""
Human assistance tool for multi-agent coordination.

Implements the LangGraph human-in-the-loop pattern for agent error handling and clarification.
"""

from langchain_core.tools import tool
from langgraph.types import interrupt

from paas_ai.utils.logging import get_logger

logger = get_logger("paas_ai.agents.tools.human_assistance")


@tool
def human_assistance(message: str) -> str:
    """
    Request assistance from a human when you need help, clarification, or guidance.

    Use this tool whenever you:
    - Need clarification on unclear requirements
    - Encounter failures or errors you can't resolve
    - Need confirmation before proceeding with important actions
    - Want to ask questions about user preferences or missing information
    - Need debugging help with technical issues
    - Are unsure how to proceed with a request

    Args:
        message: Your question, problem description, or request for help

    Returns:
        Human response with the assistance you requested
    """
    logger.info(f"[Requesting human assistance] {message}")

    human_response = interrupt({"message": message})

    return human_response["data"]

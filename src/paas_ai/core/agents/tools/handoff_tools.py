"""
Handoff tools for multi-agent coordination.

Implements the LangGraph handoff pattern for agent-to-agent communication.
"""

from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from paas_ai.utils.logging import get_logger

logger = get_logger("paas_ai.agents.tools.handoff_tools")


def create_handoff_tool(*, agent_name: str, description: str = None):
    """
    Create a handoff tool for transferring control to another agent.

    Follows the exact LangGraph pattern for handoffs.

    Args:
        agent_name: Name of the target agent
        description: Tool description (auto-generated if None)

    Returns:
        BaseTool: Handoff tool that returns Command objects
    """
    tool_name = f"transfer_to_{agent_name}"

    if description is None:
        description = f"Transfer control to the {agent_name} agent for specialized assistance."

    @tool(tool_name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Execute handoff to target agent."""
        logger.info(f"Transferring control to {agent_name}")

        # Create tool response message
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }

        # Return Command to navigate to target agent
        return Command(
            goto=agent_name,
            update={"messages": state["messages"] + [tool_message]},
            graph=Command.PARENT,
        )

    return handoff_tool


def create_handoff_tools(agent_names: list[str], current_agent: str) -> list:
    """
    Create handoff tools for all other agents.

    Args:
        agent_names: List of all available agent names
        current_agent: Name of the current agent (excluded from handoffs)

    Returns:
        List of handoff tools for all other agents
    """
    handoff_tools = []

    for agent_name in agent_names:
        if agent_name != current_agent:
            tool = create_handoff_tool(agent_name=agent_name)
            handoff_tools.append(tool)
            logger.debug(f"Created handoff tool for {current_agent} -> {agent_name}")

    return handoff_tools


# Predefined handoff tools for our agents
def create_designer_handoff():
    """Create handoff tool to Designer agent."""
    return create_handoff_tool(
        agent_name="designer",
        description="Transfer to the Designer agent for architecture design, system planning, and technical recommendations.",
    )


def create_paas_generator_handoff():
    """Create handoff tool to PaaS Manifest Generator agent."""
    return create_handoff_tool(
        agent_name="paas_manifest_generator",
        description="Transfer to the PaaS Manifest Generator agent for Cool Demo PaaS YAML configuration generation and validation.",
    )


# Create a simple handoff function for tool registry
def handoff_to_agent_tool(agent_name: str, **kwargs):
    """
    Transfer control to another specialized agent.
    Use this when you need to hand off work to a different agent with specific expertise.

    Args:
        agent_name: Name of the target agent (designer, paas_manifest_generator)
        **kwargs: Additional arguments (reason, description, etc.) - all optional

    Returns:
        Confirmation message of the handoff
    """
    # Extract reason from various possible argument names
    reason = kwargs.get("reason", "") or kwargs.get("description", "") or kwargs.get("message", "")

    # Validate agent name
    valid_agents = ["designer", "paas_manifest_generator", "supervisor"]
    if agent_name not in valid_agents:
        logger.warning(f"Invalid agent name: {agent_name}. Valid agents: {valid_agents}")
        return (
            f"Error: Invalid agent name '{agent_name}'. Valid agents are: {', '.join(valid_agents)}"
        )

    if reason:
        logger.info(f"Transferring to {agent_name}: {reason}")
        return f"Successfully transferred to {agent_name} agent: {reason}"
    else:
        logger.info(f"Transferring control to {agent_name}")
        return f"Successfully transferred to {agent_name} agent"

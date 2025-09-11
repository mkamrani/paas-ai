"""
Handoff tools for multi-agent coordination.

Implements the LangGraph handoff pattern for agent-to-agent communication.
"""

from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command

from ....utils.logging import get_logger

logger = get_logger("paas_ai.agents.handoff_tools")


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
        description="Transfer to the Designer agent for architecture design, system planning, and technical recommendations."
    )


def create_k8s_handoff():
    """Create handoff tool to K8s Manifest agent."""
    return create_handoff_tool(
        agent_name="k8s_manifest", 
        description="Transfer to the K8s Manifest agent for Kubernetes deployment configurations and YAML generation."
    ) 
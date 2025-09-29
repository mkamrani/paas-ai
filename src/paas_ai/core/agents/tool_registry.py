"""
Tool Registry for Multi-Agent System

Provides centralized tool management with name-based lookup and runtime config access.
"""

from typing import Any, Dict, List, Optional, Type

from langchain.tools import BaseTool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from paas_ai.utils.logging import get_logger

logger = get_logger("paas_ai.agents.tool_registry")


class ToolRegistry:
    """Central registry for tools used by agents."""

    _tools: Dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls, name: str, tool_class: Type[BaseTool]) -> None:
        """Register a tool class with a name."""
        cls._tools[name] = tool_class
        logger.debug(f"Registered tool: {name}")

    @classmethod
    def get_tool(cls, name: str) -> Optional[Type[BaseTool]]:
        """Get a tool class by name."""
        return cls._tools.get(name)

    @classmethod
    def create_tool(cls, name: str) -> Optional[BaseTool]:
        """Create a tool instance by name."""
        tool_class = cls.get_tool(name)
        if tool_class:
            return tool_class()
        logger.warning(f"Unknown tool: {name}")
        return None

    @classmethod
    def create_tools(cls, tool_names: List[str]) -> List[BaseTool]:
        """Create multiple tool instances from names."""
        tools = []
        for name in tool_names:
            tool = cls.create_tool(name)
            if tool:
                tools.append(tool)
        return tools

    @classmethod
    def list_tools(cls) -> List[str]:
        """List all registered tool names."""
        return list(cls._tools.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (useful for testing)."""
        cls._tools.clear()


def register_tool(name: str):
    """Decorator to register a tool class."""

    def decorator(tool_class: Type[BaseTool]):
        ToolRegistry.register(name, tool_class)
        return tool_class

    return decorator


# Function-based tool wrapper for direct function registration
class FunctionTool(BaseTool):
    """Wrapper to convert functions into LangChain tools."""

    name: str
    description: str
    func: Any
    args_schema: Optional[Any] = None

    def __init__(self, name: str, description: str, func, args_schema=None):
        super().__init__(name=name, description=description, func=func, args_schema=args_schema)

    def _run(self, *args, **kwargs):
        """Execute the wrapped function."""
        return self.func(*args, **kwargs)


def register_function_tool(name: str, description: str, args_schema=None):
    """Decorator to register a function as a tool."""

    def decorator(func):
        tool = FunctionTool(name, description, func, args_schema)
        ToolRegistry.register(name, lambda: tool)
        return func

    return decorator


# Import and register all available tools
def _register_default_tools():
    """Register all default tools."""
    try:
        # Import RAG search tool
        from .tools.rag_search import RAGSearchTool

        ToolRegistry.register("rag_search", RAGSearchTool)

        # Import file operation tools
        from .tools.file_tools import ReadFileTool, WriteFileTool

        ToolRegistry.register("write_file", WriteFileTool)
        ToolRegistry.register("read_file", ReadFileTool)

        # Note: Removed design_specification tool - Designer now creates natural language designs

        # Register PaaS manifest generation tools (structured)
        from .tools.paas_generation_tools import (
            manifest_validation_tool,
            paas_manifest_generator_tool,
        )

        class PaaSGeneratorInput(BaseModel):
            design_specification: str = Field(
                description="Structured JSON design specification. Use RAG to convert natural language design to this JSON format before calling this tool."
            )

        generator_tool = StructuredTool(
            name="paas_manifest_generator",
            description="Generate complete Cool Demo PaaS YAML manifests from a structured JSON specification. First use RAG to understand platform capabilities, then map natural language design to JSON format, then call this tool.",
            func=paas_manifest_generator_tool,
            args_schema=PaaSGeneratorInput,
        )
        ToolRegistry.register("paas_manifest_generator", lambda: generator_tool)

        class ManifestValidationInput(BaseModel):
            manifest_data: str = Field(
                description="Generated manifest data as JSON string to validate"
            )

        validation_tool = StructuredTool(
            name="manifest_validation",
            description="Validate generated PaaS manifests for completeness and correctness. Use this tool to check your generated configurations before final delivery.",
            func=manifest_validation_tool,
            args_schema=ManifestValidationInput,
        )
        ToolRegistry.register("manifest_validation", lambda: validation_tool)

        # Register handoff tool
        from .tools.handoff_tools import handoff_to_agent_tool

        class HandoffInput(BaseModel):
            agent_name: str = Field(
                description="Name of the target agent. Must be one of: designer, paas_manifest_generator, supervisor"
            )
            reason: str = Field(default="", description="Optional reason for the handoff")

        handoff_tool = StructuredTool(
            name="handoff_to_agent",
            description="Transfer control to another specialized agent. Use this when you need to hand off work to a different agent with specific expertise. Available agents: designer (for architecture design), paas_manifest_generator (for YAML generation), supervisor (for coordination).",
            func=handoff_to_agent_tool,
            args_schema=HandoffInput,
        )
        ToolRegistry.register("handoff_to_agent", lambda: handoff_tool)

        from .tools.human_assistance_tools import human_assistance

        # Register the decorated tool directly
        ToolRegistry.register("human_assistance", lambda: human_assistance)

        logger.info(f"Registered {len(ToolRegistry.list_tools())} default tools")

    except ImportError as e:
        logger.warning(f"Could not import some tools: {e}")


# Tool configurations for different agents
AGENT_TOOL_CONFIGS = {
    "designer": ["rag_search", "handoff_to_agent", "human_assistance"],
    "paas_manifest_generator": [
        "rag_search",
        "paas_manifest_generator",
        "manifest_validation",
        "write_file",
        "read_file",
        "handoff_to_agent",
        "human_assistance",
    ],
    "supervisor": ["handoff_to_agent", "human_assistance"],
}


def get_tools_for_agent(agent_name: str) -> List[BaseTool]:
    """Get the configured tools for a specific agent."""
    tool_names = AGENT_TOOL_CONFIGS.get(agent_name, [])
    return ToolRegistry.create_tools(tool_names)


# Register default tools on module import
_register_default_tools()

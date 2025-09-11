"""
Tool Registry for Multi-Agent System

Provides centralized tool management with name-based lookup and runtime config access.
"""

from typing import Dict, Type, List, Optional, Any
from langchain.tools import BaseTool
from ...utils.logging import get_logger

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


# Import and register all available tools
def _register_default_tools():
    """Register all default tools."""
    try:
        # Import existing tools
        from .tools.rag_search import RAGSearchTool
        ToolRegistry.register("search_knowledge_base", RAGSearchTool)
        
        # Import design tools
        from .tools.design_tools import MermaidDiagramTool, ArchitecturePatternTool
        ToolRegistry.register("generate_mermaid_diagram", MermaidDiagramTool)
        ToolRegistry.register("explain_architecture_pattern", ArchitecturePatternTool)
        
        # Import K8s tools
        from .tools.k8s_tools import GenerateManifestTool, ValidateManifestTool
        ToolRegistry.register("generate_k8s_manifest", GenerateManifestTool)
        ToolRegistry.register("validate_k8s_manifest", ValidateManifestTool)
        
        logger.info(f"Registered {len(ToolRegistry.list_tools())} default tools")
        
    except ImportError as e:
        logger.warning(f"Could not import some tools: {e}")


# Register default tools on module import
_register_default_tools() 
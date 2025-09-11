"""
Agents module with multi-agent RAG integration using LangGraph.
"""

# Import the new multi-agent system as the primary interface
from .multi_agent_system import MultiAgentSystem

# For backward compatibility, also expose as RAGAgent
RAGAgent = MultiAgentSystem

# Also export the individual components for direct access
from .base_agent import BaseAgent
from .tool_registry import ToolRegistry
from .tools import RAGSearchTool

__all__ = [
    'RAGAgent',           # Alias to MultiAgentSystem for backward compatibility
    'MultiAgentSystem',   # New multi-agent system
    'BaseAgent',          # Individual agent wrapper
    'ToolRegistry',       # Tool management
    'RAGSearchTool'       # Legacy tool export
]

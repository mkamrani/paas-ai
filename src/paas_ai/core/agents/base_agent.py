"""
BaseAgent - Thin wrapper around create_react_agent with multi-agent capabilities.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .tool_registry import ToolRegistry
# Removed unused import
from ..config import Config
from ...utils.logging import get_logger


class BaseAgent:
    """
    Thin wrapper around create_react_agent with mode/tool/prompt management.
    
    Supports both supervisor and swarm modes with automatic handoff tool generation.
    """
    
    def __init__(
        self,
        name: str,
        tool_names: List[str],
        config: Config,
        mode: Literal["supervisor", "swarm"] = "supervisor",
        available_agents: Optional[List[str]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name (used for prompts and identification)
            tool_names: List of tool names to use (from ToolRegistry)
            config: Configuration object
            mode: Operation mode (supervisor or swarm)
            available_agents: List of available agents for handoff generation
        """
        self.name = name
        self.tool_names = tool_names.copy()
        self.config = config
        self.mode = mode
        self.available_agents = available_agents or []
        self.logger = get_logger(f"paas_ai.agents.{name}")
        
        # Add handoff tools for swarm mode
        if mode == "swarm" and self.available_agents:
            self._add_handoff_tool_names()
        
        # Load prompt template
        self.prompt = self._load_prompt_template()
        
        # Create tools from names
        self.tools = self._create_tools_from_names()
        
        # Get model for this agent
        self.model = self._get_model()
        
        # Create the react agent (compiled graph)
        self.react_agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=self.prompt,
            name=self.name
        )
        
        self.logger.info(f"Initialized {name} agent in {mode} mode with {len(self.tools)} tools")
    
    def _add_handoff_tool_names(self) -> None:
        """Add handoff tool names for swarm mode."""
        for agent_name in self.available_agents:
            if agent_name != self.name:
                handoff_tool_name = f"transfer_to_{agent_name}"
                if handoff_tool_name not in self.tool_names:
                    self.tool_names.append(handoff_tool_name)
                    self.logger.debug(f"Added handoff tool: {handoff_tool_name}")
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from prompts/{agent_name}/system.md"""
        prompt_path = Path(__file__).parent / "prompts" / self.name / "system.md"
        
        try:
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8")
            else:
                # Fallback to default prompt
                self.logger.warning(f"Prompt file not found: {prompt_path}, using default")
                return self._get_default_prompt()
        except Exception as e:
            self.logger.error(f"Error loading prompt template: {e}")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Get default prompt if specific prompt file is not found."""
        return f"""You are a helpful AI assistant specialized in {self.name.replace('_', ' ')} tasks.

Use the available tools to help users with their requests. Be thorough in your research and provide detailed, actionable responses.

When you need specialized help from another domain, don't hesitate to transfer to the appropriate agent."""
    
    def _create_tools_from_names(self) -> List[Any]:
        """Create tool instances from names using registry and handoff tools."""
        tools = []
        
        for tool_name in self.tool_names:
            if tool_name.startswith("transfer_to_"):
                # Create handoff tool
                target_agent = tool_name.replace("transfer_to_", "")
                handoff_tool = self._create_handoff_tool(target_agent)
                if handoff_tool:
                    tools.append(handoff_tool)
            else:
                # Get tool from registry
                tool = ToolRegistry.create_tool(tool_name)
                if tool:
                    tools.append(tool)
                else:
                    self.logger.warning(f"Unknown tool: {tool_name}")
        
        return tools
    
    def _create_handoff_tool(self, target_agent: str):
        """Create a handoff tool for the target agent."""
        try:
            from .tools.handoff_tools import create_handoff_tool
            return create_handoff_tool(agent_name=target_agent)
        except Exception as e:
            self.logger.error(f"Error creating handoff tool for {target_agent}: {e}")
            return None
    
    def _get_model(self) -> BaseChatModel:
        """Get LLM model for this agent from config."""
        # Get agent-specific config or use default
        agent_config = self.config.agents.get(self.name, {})
        model_name = agent_config.get("model", "gpt-4o-mini")
        temperature = agent_config.get("temperature", 0.1)
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it with: export OPENAI_API_KEY='your-key-here'"
            )
        
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
    
    def invoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke the agent with state and optional config."""
        return self.react_agent.invoke(state, config)
    
    def stream(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Stream responses from the agent."""
        return self.react_agent.stream(state, config)
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get information about available tools."""
        tool_info = []
        for tool in self.tools:
            info = {
                "name": tool.name,
                "description": tool.description,
            }
            if hasattr(tool, 'args_schema') and tool.args_schema:
                info["args_schema"] = tool.args_schema.schema()
            tool_info.append(info)
        return tool_info
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for this agent."""
        agent_config = self.config.agents.get(self.name, {})
        return {
            "name": self.name,
            "mode": self.mode,
            "model": agent_config.get("model", "gpt-4o-mini"),
            "temperature": agent_config.get("temperature", 0.1),
            "tools": [tool.name for tool in self.tools],
            "available_handoffs": [
                name for name in self.available_agents if name != self.name
            ] if self.mode == "swarm" else []
        } 
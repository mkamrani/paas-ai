"""
BaseAgent - Thin wrapper around create_react_agent with multi-agent capabilities.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .tool_registry import ToolRegistry
# Removed unused import
from ..config import Config
from paas_ai.utils.logging import get_logger


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
            api_key=api_key,
            stream_usage=True  # Enable token usage tracking for both streaming and regular calls
        )
    
    def invoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke the agent with state and optional config."""
        start_time = time.time()
        
        # Invoke the agent
        result = self.react_agent.invoke(state, config)
        
        # Extract and track token usage if enabled
        if self._should_track_tokens(config):
            processing_time = time.time() - start_time
            self._extract_and_track_tokens(result, config, processing_time)
        
        return result
    
    def stream(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Stream responses from the agent."""
        if not self._should_track_tokens(config):
            # Simple pass-through if not tracking
            yield from self.react_agent.stream(state, config)
            return
        
        # Streaming with token tracking
        start_time = time.time()
        accumulated_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        chunk_count = 0
        
        for chunk in self.react_agent.stream(state, config):
            # Extract token info from each chunk
            chunk_tokens = self._extract_chunk_tokens(chunk)
            if chunk_tokens:
                accumulated_tokens["input_tokens"] += chunk_tokens.get("input_tokens", 0)
                accumulated_tokens["output_tokens"] += chunk_tokens.get("output_tokens", 0)
                accumulated_tokens["total_tokens"] += chunk_tokens.get("total_tokens", 0)
            
            chunk_count += 1
            yield chunk
        
        # Track final accumulated tokens
        if accumulated_tokens["total_tokens"] > 0:
            processing_time = time.time() - start_time
            self._track_accumulated_tokens(accumulated_tokens, config, processing_time)
    
    def _should_track_tokens(self, config: Optional[Dict[str, Any]]) -> bool:
        """Check if token tracking is enabled."""
        if not config:
            return False
        
        configurable = config.get('configurable', {})
        token_tracker = configurable.get('token_tracker')
        return token_tracker and token_tracker.enabled
    
    def _extract_and_track_tokens(self, result: Dict[str, Any], config: Dict[str, Any], processing_time: float) -> None:
        """Extract token usage from result and track it."""
        usage_data = self._extract_token_usage(result)
        if usage_data:
            self._track_tokens(usage_data, config, processing_time)
    
    def _extract_chunk_tokens(self, chunk: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Extract token usage from a streaming chunk."""
        # Check if chunk contains usage metadata
        if hasattr(chunk, 'usage_metadata'):
            usage = chunk.usage_metadata
            return {
                'input_tokens': getattr(usage, 'input_tokens', 0),
                'output_tokens': getattr(usage, 'output_tokens', 0),
                'total_tokens': getattr(usage, 'total_tokens', 0)
            }
        
        # Check for usage in chunk data
        if isinstance(chunk, dict) and 'usage' in chunk:
            usage = chunk['usage']
            return {
                'input_tokens': usage.get('prompt_tokens', 0),
                'output_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
        
        return None
    
    def _track_accumulated_tokens(self, accumulated_tokens: Dict[str, int], config: Dict[str, Any], processing_time: float) -> None:
        """Track accumulated tokens from streaming."""
        self._track_tokens(accumulated_tokens, config, processing_time)
    
    def _track_tokens(self, usage_data: Dict[str, int], config: Dict[str, Any], processing_time: float) -> None:
        """Track token usage with the session tracker."""
        configurable = config.get('configurable', {})
        token_tracker = configurable.get('token_tracker')
        
        if token_tracker:
            # Get agent-specific config for model name
            agent_config = self.config.agents.get(self.name, {})
            model_name = agent_config.get("model", "gpt-4o-mini")
            
            # Generate request ID if not provided
            request_id = configurable.get('request_id', f"{self.name}-{int(time.time())}")
            
            token_tracker.track(
                usage_data=usage_data,
                model=model_name,
                agent=self.name,
                request_id=request_id,
                processing_time=processing_time
            )
    
    def _extract_token_usage(self, result: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Extract token usage from LLM response."""
        # Check for token usage in various possible locations
        messages = result.get("messages", [])
        
        # Debug logging to understand the structure (use info level in verbose mode)
        if hasattr(self, 'config') and self.config.multi_agent.verbose:
            self.logger.info(f"🔍 Token extraction - Result keys: {list(result.keys())}")
            self.logger.info(f"🔍 Token extraction - Messages count: {len(messages)}")
            if messages:
                last_msg = messages[-1]
                self.logger.info(f"🔍 Token extraction - Last message type: {type(last_msg)}")
                if hasattr(last_msg, 'usage_metadata'):
                    self.logger.info(f"🔍 Found usage_metadata: {last_msg.usage_metadata}")
                if hasattr(last_msg, 'response_metadata'):
                    self.logger.info(f"🔍 Found response_metadata keys: {list(last_msg.response_metadata.keys())}")
                    if 'usage' in last_msg.response_metadata:
                        self.logger.info(f"🔍 Found usage in response_metadata: {last_msg.response_metadata['usage']}")
                    if 'token_usage' in last_msg.response_metadata:
                        self.logger.info(f"🔍 Found token_usage in response_metadata: {last_msg.response_metadata['token_usage']}")
        
        for message in reversed(messages):  # Check latest first
            # Check for usage_metadata (LangChain v0.2+)
            if hasattr(message, 'usage_metadata') and message.usage_metadata:
                usage = message.usage_metadata
                if hasattr(self, 'config') and self.config.multi_agent.verbose:
                    self.logger.info(f"🔍 Extracting from usage_metadata: {usage}")
                # usage_metadata is a dictionary, not an object
                return {
                    'input_tokens': usage.get('input_tokens', 0),
                    'output_tokens': usage.get('output_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }
            
            # Check additional_kwargs for usage info (older LangChain versions)
            if hasattr(message, 'additional_kwargs'):
                usage = message.additional_kwargs.get('usage')
                if usage:
                    if hasattr(self, 'config') and self.config.multi_agent.verbose:
                        self.logger.debug(f"🔍 Found additional_kwargs usage: {usage}")
                    return {
                        'input_tokens': usage.get('prompt_tokens', 0),
                        'output_tokens': usage.get('completion_tokens', 0),
                        'total_tokens': usage.get('total_tokens', 0)
                    }
            
            # Check for response_metadata
            if hasattr(message, 'response_metadata'):
                metadata = message.response_metadata
                
                # Check for token_usage field (newer format)
                if 'token_usage' in metadata:
                    usage = metadata['token_usage']
                    if hasattr(self, 'config') and self.config.multi_agent.verbose:
                        self.logger.info(f"🔍 Extracting from response_metadata token_usage: {usage}")
                    return {
                        'input_tokens': usage.get('prompt_tokens', 0),
                        'output_tokens': usage.get('completion_tokens', 0),
                        'total_tokens': usage.get('total_tokens', 0)
                    }
                
                # Check for usage field (older format)
                if 'usage' in metadata:
                    usage = metadata['usage']
                    if hasattr(self, 'config') and self.config.multi_agent.verbose:
                        self.logger.info(f"🔍 Extracting from response_metadata usage: {usage}")
                    return {
                        'input_tokens': usage.get('prompt_tokens', 0),
                        'output_tokens': usage.get('completion_tokens', 0),
                        'total_tokens': usage.get('total_tokens', 0)
                    }
        
        # Check top-level result for usage info
        if 'usage' in result:
            usage = result['usage']
            if hasattr(self, 'config') and self.config.multi_agent.verbose:
                self.logger.debug(f"🔍 Found top-level usage: {usage}")
            return {
                'input_tokens': usage.get('prompt_tokens', 0),
                'output_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
        
        if hasattr(self, 'config') and self.config.multi_agent.verbose:
            self.logger.debug("🔍 No token usage found in response")
        
        return None
    
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
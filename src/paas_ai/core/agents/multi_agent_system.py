"""
Multi-Agent System - Main coordinator for specialized agents.

Replaces the single RAGAgent with a system that can operate in supervisor or swarm mode.
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from paas_ai.utils.logging import get_logger

from ..config import Config
from .base_agent import BaseAgent
from .persistence import create_checkpointer, generate_thread_id
from .token_tracking import SessionTokenTracker, TokenCallbackFactory
from .tool_registry import AGENT_TOOL_CONFIGS, ToolRegistry


class MultiAgentSystem:
    """
    Multi-Agent System coordinator

    Manages specialized agents in either supervisor or swarm mode
    """

    def __init__(self, config: Config):
        """Initialize the multi-agent system."""
        self.config = config
        self.mode = config.multi_agent.mode
        self.verbose = config.multi_agent.verbose
        self.logger = get_logger("paas_ai.multi_agent_system")

        # Initialize token tracking with verbosity
        self.token_tracker = self._initialize_token_tracker()

        # Initialize persistence
        self.checkpointer = create_checkpointer(config.multi_agent.persistence)

        # Initialize specialized agents
        self.agents = self._initialize_agents()

        # Build coordination system based on mode
        self.coordinator = self._build_coordinator()

        # Log initialization (respects verbosity)
        if self.verbose:
            self.logger.info(f"🤖 MultiAgentSystem initialized in {self.mode} mode")
            self.logger.info(
                f"📊 Token tracking: {'enabled' if config.multi_agent.track_tokens else 'disabled'}"
            )
            self.logger.info(f"🔊 Verbose mode: enabled")
            if config.multi_agent.token_callback:
                self.logger.info(f"🔗 Token callback: {config.multi_agent.token_callback}")
        else:
            self.logger.info(
                f"MultiAgentSystem initialized in {self.mode} mode with {len(self.agents)} agents"
            )

    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize all specialized agents."""
        agents = {}
        available_agent_names = ["designer", "paas_manifest_generator"]

        # Designer Agent
        designer_tools = AGENT_TOOL_CONFIGS.get(
            "designer", ["rag_search", "design_specification", "handoff_to_agent"]
        )
        agents["designer"] = BaseAgent(
            name="designer",
            tool_names=designer_tools,
            config=self.config,
            mode=self.mode,
            available_agents=available_agent_names,
        )

        # PaaS Manifest Generator Agent
        paas_tools = AGENT_TOOL_CONFIGS.get(
            "paas_manifest_generator",
            ["rag_search", "paas_manifest_generator", "manifest_validation", "handoff_to_agent"],
        )
        agents["paas_manifest_generator"] = BaseAgent(
            name="paas_manifest_generator",
            tool_names=paas_tools,
            config=self.config,
            mode=self.mode,
            available_agents=available_agent_names,
        )

        return agents

    def _initialize_token_tracker(self) -> SessionTokenTracker:
        """Initialize token tracking system with callback support."""
        config = self.config.multi_agent

        # Create callback if specified
        callback = None
        if config.track_tokens and config.token_callback:
            # Create callback with appropriate parameters based on type
            callback_name = config.token_callback
            if callback_name == "console":
                callback = TokenCallbackFactory.create_callback(callback_name, verbose=self.verbose)
            elif callback_name == "json_file":
                callback = TokenCallbackFactory.create_callback(
                    callback_name, file_path="token_usage.jsonl"
                )
            elif callback_name == "webhook":
                # Webhook requires URL - skip if not configured properly
                self.logger.warning("Webhook callback requires webhook_url configuration")
                callback = None
            else:
                # Try to create with no extra parameters
                callback = TokenCallbackFactory.create_callback(callback_name)

            if callback is None:
                self.logger.warning(f"Failed to create token callback: {config.token_callback}")

        return SessionTokenTracker(
            enabled=config.track_tokens, callback=callback, verbose=self.verbose
        )

    def _build_coordinator(self):
        """Build coordination system based on mode."""
        if self.mode == "supervisor":
            return self._build_supervisor()
        elif self.mode == "swarm":
            return self._build_swarm()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _build_supervisor(self):
        """Build supervisor-based coordination."""
        try:
            # Try to import and use langgraph-supervisor
            from langgraph_supervisor import create_supervisor

            # Get model for supervisor
            supervisor_model = self._get_supervisor_model()

            # Load supervisor prompt
            supervisor_prompt = self._load_supervisor_prompt()

            # Create supervisor with our agents
            supervisor = create_supervisor(
                agents=[agent.react_agent for agent in self.agents.values()],
                model=supervisor_model,
                prompt=supervisor_prompt,
            ).compile(checkpointer=self.checkpointer)

            self.logger.info("Supervisor coordination system built successfully")
            return supervisor

        except ImportError as e:
            error_msg = (
                "Supervisor mode requires the 'langgraph-supervisor' package. "
                "Install it with: poetry add langgraph-supervisor"
            )
            self.logger.error(error_msg)
            raise ImportError(error_msg) from e

    def _build_swarm(self):
        """Build swarm-based coordination."""
        try:
            # Try to import and use langgraph-swarm
            from langgraph_swarm import create_swarm

            # Create swarm with our agents
            swarm = create_swarm(
                agents=[agent.react_agent for agent in self.agents.values()],
                default_active_agent=self.config.multi_agent.default_agent,
            ).compile(checkpointer=self.checkpointer)

            self.logger.info("Swarm coordination system built successfully")
            return swarm

        except ImportError as e:
            error_msg = (
                "Swarm mode requires the 'langgraph-swarm' package. "
                "Install it with: poetry add langgraph-swarm"
            )
            self.logger.error(error_msg)
            raise ImportError(error_msg) from e

    def _get_supervisor_model(self) -> ChatOpenAI:
        """Get model for supervisor coordination."""
        # Use same model as default agent
        agent_config = self.config.agents.get("designer", {})
        model_name = agent_config.get("model", "gpt-4o-mini")
        temperature = agent_config.get("temperature", 0.1)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

    def _load_supervisor_prompt(self) -> str:
        """Load supervisor prompt template."""
        from pathlib import Path

        prompt_path = Path(__file__).parent / "prompts" / "supervisor" / "system.md"

        try:
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8")
            else:
                # Fallback prompt
                return self._get_default_supervisor_prompt()
        except Exception as e:
            self.logger.error(f"Error loading supervisor prompt: {e}")
            return self._get_default_supervisor_prompt()

    def _get_default_supervisor_prompt(self) -> str:
        """Get default supervisor prompt."""
        return """You are a multi-agent coordinator for a Cool Demo PaaS AI system. 

You manage two specialized agents:
- designer: Handles infrastructure architecture, AWS service selection, and high-level design
- paas_manifest_generator: Handles Cool Demo PaaS YAML generation, configuration implementation, and validation

Analyze user requests and route them to the appropriate specialist agent based on the content:
- Architecture, design, service selection, requirements analysis → designer
- YAML generation, configuration implementation, manifest creation → paas_manifest_generator

For complex requests that span both domains, start with the designer agent for architectural planning, then route to the generator for implementation once you have complete design specifications.

The standard workflow is: Designer creates specifications → Generator creates YAML configurations

Assign work to one agent at a time, do not call agents in parallel.
Do not do any work yourself.
"""

    # === Public API (same as RAGAgent for backward compatibility) ===

    def chat(self, messages: List[BaseMessage], thread_id: Optional[str] = None) -> str:
        """
        Chat with conversation history - same interface as RAGAgent.

        Args:
            messages: List of conversation messages
            thread_id: Optional thread ID for conversation persistence

        Returns:
            The agent's response
        """
        # Generate thread ID if not provided
        if thread_id is None:
            thread_id = generate_thread_id()

        if self.verbose:
            self.logger.info(
                f"💬 Processing chat with {len(messages)} messages (thread: {thread_id})"
            )

        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            # Create initial state with conversation history
            initial_state = {"messages": messages}

            # Add runtime config with token tracker and thread ID
            runtime_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "paas_config": self.config,
                    "token_tracker": self.token_tracker,
                    "request_id": request_id,
                    "recursion_limit": 100,  # Increase from default 25 to handle complex multi-agent workflows
                    "subgraphs": True,
                },
            }

            # Run the coordination system
            result = self.coordinator.invoke(initial_state, config=runtime_config)

            # Extract response
            response = self._extract_response(result)

            # Show timing and token info in verbose mode
            if self.verbose:
                duration = time.time() - start_time
                self.logger.info(f"⏱️ Chat response generated in {duration:.2f}s")

                # Show token summary if tracking enabled
                if self.config.multi_agent.track_tokens:
                    session_summary = self.token_tracker.get_last_request_summary()
                    if session_summary.get("total_tokens", 0) > 0:
                        tokens = session_summary["total_tokens"]
                        agent = session_summary.get("agent", "unknown")
                        model = session_summary.get("model", "unknown")
                        self.logger.info(f"🪙 Token usage: {tokens} tokens ({agent} using {model})")

            return response

        except Exception as e:
            if self.verbose:
                self.logger.error(f"❌ Error in chat: {e}")
            error_msg = f"Error in chat: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def chat_stream(self, messages: List[BaseMessage], thread_id: Optional[str] = None):
        """
        Chat with conversation history and streaming response.

        Args:
            messages: List of conversation messages
            thread_id: Optional thread ID for conversation persistence

        Yields:
            str: Streaming tokens from the LLM
        """
        # Generate thread ID if not provided
        if thread_id is None:
            thread_id = generate_thread_id()

        if self.verbose:
            self.logger.info(
                f"💬 Processing chat with streaming: {len(messages)} messages (thread: {thread_id})"
            )

        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            # Create initial state with conversation history
            initial_state = {"messages": messages}

            # Add runtime config with token tracker and thread ID
            runtime_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "paas_config": self.config,
                    "token_tracker": self.token_tracker,
                    "request_id": request_id,
                },
                "recursion_limit": 50,  # Increase from default 25 to handle complex multi-agent workflows
            }

            # Stream tokens from the coordination system using "messages" mode
            for token, metadata in self.coordinator.stream(
                initial_state, config=runtime_config, stream_mode="messages"
            ):
                # token is the actual LLM token, metadata contains node info
                if hasattr(token, "content") and token.content:
                    yield token.content
                elif isinstance(token, str):
                    yield token

            # Show timing info in verbose mode after streaming completes
            if self.verbose:
                duration = time.time() - start_time
                self.logger.info(f"⏱️ Chat streaming completed in {duration:.2f}s")

                # Show token summary if tracking enabled
                if self.config.multi_agent.track_tokens:
                    session_summary = self.token_tracker.get_last_request_summary()
                    if session_summary.get("total_tokens", 0) > 0:
                        tokens = session_summary["total_tokens"]
                        agent = session_summary.get("agent", "unknown")
                        model = session_summary.get("model", "unknown")
                        self.logger.info(f"🪙 Token usage: {tokens} tokens ({agent} using {model})")

        except Exception as e:
            if self.verbose:
                self.logger.error(f"❌ Error in chat streaming: {e}")
            error_msg = f"Error in chat streaming: {str(e)}"
            self.logger.error(error_msg)
            # Yield error as final chunk
            yield f"\n❌ {error_msg}"

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools across agents."""
        all_tools = []
        for agent in self.agents.values():
            agent_tools = agent.get_available_tools()
            for tool in agent_tools:
                # Add agent context to tool info
                tool_with_agent = tool.copy()
                tool_with_agent["agent"] = agent.name
                all_tools.append(tool_with_agent)
        return all_tools

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary including multi-agent info."""
        base_summary = {
            "llm": {
                "provider": self.config.llm.provider,
                "model": self.config.llm.model_name,
                "temperature": self.config.llm.temperature,
            },
            "embedding": {
                "type": self.config.embedding.type,
                "model": self.config.embedding.model_name,
            },
            "vectorstore": {
                "type": self.config.vectorstore.type,
                "collection": self.config.vectorstore.collection_name,
                "directory": str(self.config.vectorstore.persist_directory),
            },
            "multi_agent": {
                "enabled": True,
                "mode": self.mode,
                "agents": list(self.agents.keys()),
                "default_agent": self.config.multi_agent.default_agent,
                "track_tokens": self.config.multi_agent.track_tokens,
                "verbose": self.config.multi_agent.verbose,
                "token_callback": self.config.multi_agent.token_callback,
            },
        }

        # Add per-agent summaries
        base_summary["agent_details"] = {}
        for name, agent in self.agents.items():
            base_summary["agent_details"][name] = agent.get_config_summary()

        return base_summary

    def get_token_session_summary(self) -> Dict[str, Any]:
        """
        Get current token usage session summary.

        Returns:
            Dictionary with token usage information including totals,
            per-agent breakdown, and session statistics.
        """
        return self.token_tracker.get_session_summary()

    def get_last_token_summary(self) -> Dict[str, Any]:
        """
        Get token summary for the last request.

        Returns:
            Dictionary with token usage for the most recent operation.
        """
        return self.token_tracker.get_last_request_summary()

    def clear_token_history(self) -> None:
        """Clear the token usage history for this session."""
        self.token_tracker.clear()

    def _extract_response(self, result: Dict[str, Any]) -> str:
        """Extract the final response from coordination result."""
        messages = result.get("messages", [])
        if not messages:
            return "No response generated"

        final_message = messages[-1]

        if hasattr(final_message, "content"):
            return final_message.content
        else:
            return str(final_message)

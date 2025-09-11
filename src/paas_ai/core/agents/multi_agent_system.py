"""
Multi-Agent System - Main coordinator for specialized agents.

Replaces the single RAGAgent with a system that can operate in supervisor or swarm mode.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI

from .base_agent import BaseAgent
from .tool_registry import ToolRegistry
from ..config import Config
from ...utils.logging import get_logger


class MultiAgentSystem:
    """
    Multi-Agent System coordinator - replaces RAGAgent.
    
    Manages specialized agents in either supervisor or swarm mode while maintaining
    the same interface as the original RAGAgent for backward compatibility.
    """
    
    def __init__(self, config: Config):
        """Initialize the multi-agent system."""
        self.config = config
        self.mode = config.multi_agent.mode
        self.logger = get_logger("paas_ai.multi_agent_system")
        
        # Initialize specialized agents
        self.agents = self._initialize_agents()
        
        # Build coordination system based on mode
        self.coordinator = self._build_coordinator()
        
        self.logger.info(f"MultiAgentSystem initialized in {self.mode} mode with {len(self.agents)} agents")
    
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize all specialized agents."""
        agents = {}
        available_agent_names = ["designer", "k8s_manifest"]
        
        # Designer Agent
        designer_tools = [
            "search_knowledge_base",
            "generate_mermaid_diagram",
            "explain_architecture_pattern"
        ]
        agents["designer"] = BaseAgent(
            name="designer",
            tool_names=designer_tools,
            config=self.config,
            mode=self.mode,
            available_agents=available_agent_names
        )
        
        # K8s Manifest Agent
        k8s_tools = [
            "search_knowledge_base",
            "generate_k8s_manifest",
            "validate_k8s_manifest"
        ]
        agents["k8s_manifest"] = BaseAgent(
            name="k8s_manifest",
            tool_names=k8s_tools,
            config=self.config,
            mode=self.mode,
            available_agents=available_agent_names
        )
        
        return agents
    
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
                prompt=supervisor_prompt
            ).compile()
            
            self.logger.info("Supervisor coordination system built successfully")
            return supervisor
            
        except ImportError:
            self.logger.warning("langgraph-supervisor not available, falling back to simple routing")
            return self._build_simple_router()
    
    def _build_swarm(self):
        """Build swarm-based coordination."""
        try:
            # Try to import and use langgraph-swarm
            from langgraph_swarm import create_swarm
            
            # Create swarm with our agents
            swarm = create_swarm(
                agents=[agent.react_agent for agent in self.agents.values()],
                default_active_agent=self.config.multi_agent.default_agent
            ).compile()
            
            self.logger.info("Swarm coordination system built successfully")
            return swarm
            
        except ImportError:
            self.logger.warning("langgraph-swarm not available, falling back to simple routing")
            return self._build_simple_router()
    
    def _build_simple_router(self):
        """Build simple routing fallback when langgraph packages aren't available."""
        self.logger.info("Using simple routing fallback")
        return SimpleRouter(self.agents, self.config.multi_agent.default_agent)
    
    def _get_supervisor_model(self) -> ChatOpenAI:
        """Get model for supervisor coordination."""
        # Use same model as default agent
        agent_config = self.config.agents.get("designer", {})
        model_name = agent_config.get("model", "gpt-4o-mini")
        temperature = agent_config.get("temperature", 0.1)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
    
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
        return """You are a multi-agent coordinator for a PaaS AI system. 

You manage two specialized agents:
- designer: Handles system architecture, design patterns, and Mermaid diagrams
- k8s_manifest: Handles Kubernetes deployments, YAML generation, and container orchestration

Analyze user requests and route them to the appropriate specialist agent based on the content:
- Architecture, design, patterns, diagrams → designer
- Kubernetes, deployment, YAML, containers → k8s_manifest

For complex requests that span both domains, start with the designer agent for architectural planning, then let the workflow continue naturally."""
    
    # === Public API (same as RAGAgent for backward compatibility) ===
    
    def ask(self, question: str) -> str:
        """
        Ask a question - same interface as RAGAgent.
        
        Args:
            question: The question to ask
            
        Returns:
            The agent's response
        """
        self.logger.info(f"Processing question in {self.mode} mode: {question[:100]}...")
        
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=question)]
            }
            
            # Add runtime config
            runtime_config = {
                "configurable": {
                    "paas_config": self.config
                }
            }
            
            # Run the coordination system
            result = self.coordinator.invoke(initial_state, config=runtime_config)
            
            # Extract response
            return self._extract_response(result)
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def chat(self, messages: List[BaseMessage]) -> str:
        """
        Chat with conversation history - same interface as RAGAgent.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            The agent's response
        """
        try:
            # Create initial state with conversation history
            initial_state = {"messages": messages}
            
            # Add runtime config
            runtime_config = {
                "configurable": {
                    "paas_config": self.config
                }
            }
            
            # Run the coordination system
            result = self.coordinator.invoke(initial_state, config=runtime_config)
            
            # Extract response
            return self._extract_response(result)
            
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
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
                "temperature": self.config.llm.temperature
            },
            "embedding": {
                "type": self.config.embedding.type,
                "model": self.config.embedding.model_name
            },
            "vectorstore": {
                "type": self.config.vectorstore.type,
                "collection": self.config.vectorstore.collection_name,
                "directory": str(self.config.vectorstore.persist_directory)
            },
            "multi_agent": {
                "enabled": True,
                "mode": self.mode,
                "agents": list(self.agents.keys()),
                "default_agent": self.config.multi_agent.default_agent
            }
        }
        
        # Add per-agent summaries
        base_summary["agent_details"] = {}
        for name, agent in self.agents.items():
            base_summary["agent_details"][name] = agent.get_config_summary()
        
        return base_summary
    
    def _extract_response(self, result: Dict[str, Any]) -> str:
        """Extract the final response from coordination result."""
        messages = result.get("messages", [])
        if not messages:
            return "No response generated"
        
        final_message = messages[-1]
        
        if hasattr(final_message, 'content'):
            return final_message.content
        else:
            return str(final_message)


class SimpleRouter:
    """
    Simple fallback router when langgraph coordination packages aren't available.
    
    Routes requests to appropriate agents based on keywords and returns responses.
    """
    
    def __init__(self, agents: Dict[str, BaseAgent], default_agent: str):
        """Initialize simple router."""
        self.agents = agents
        self.default_agent = default_agent
        self.logger = get_logger("paas_ai.simple_router")
    
    def invoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route request to appropriate agent."""
        messages = state.get("messages", [])
        if not messages:
            return state
        
        # Get the user's message
        user_message = None
        for msg in reversed(messages):
            if hasattr(msg, 'content') and isinstance(msg, HumanMessage):
                user_message = msg.content.lower()
                break
        
        if not user_message:
            return state
        
        # Simple keyword-based routing
        agent_name = self._route_by_keywords(user_message)
        
        self.logger.info(f"Routing to {agent_name} agent")
        
        # Invoke the selected agent
        selected_agent = self.agents[agent_name]
        result = selected_agent.invoke(state, config)
        
        return result
    
    def _route_by_keywords(self, message: str) -> str:
        """Route based on keywords in the message."""
        # K8s/deployment keywords
        k8s_keywords = [
            "kubernetes", "k8s", "kubectl", "deployment", "service", "ingress",
            "yaml", "manifest", "pod", "container", "docker", "deploy",
            "scale", "autoscaling", "configmap", "secret", "namespace"
        ]
        
        # Design/architecture keywords
        design_keywords = [
            "architecture", "design", "pattern", "mermaid", "diagram",
            "microservice", "serverless", "event-driven", "system",
            "scalability", "technology", "stack", "structure"
        ]
        
        # Count keyword matches
        k8s_score = sum(1 for keyword in k8s_keywords if keyword in message)
        design_score = sum(1 for keyword in design_keywords if keyword in message)
        
        # Route based on highest score
        if k8s_score > design_score:
            return "k8s_manifest"
        elif design_score > 0:
            return "designer"
        else:
            # Default routing
            return self.default_agent 
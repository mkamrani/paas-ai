"""
RAG Agent using LangGraph for workflow orchestration.
"""

import os
from typing import List, Dict, Any, Optional
from functools import partial

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .nodes import AgentState, call_model, call_tools, should_continue
from .tools import RAGSearchTool
from ..config import Config
from ...utils.logging import get_logger


class RAGAgent:
    """
    Basic RAG agent that can answer questions using indexed knowledge.
    """
    
    def __init__(self, config: Config):
        """Initialize the RAG agent."""
        self.config = config
        self.logger = get_logger("paas_ai.agent.rag")
        
        # Initialize LLM based on config
        self.model = self._initialize_llm()
        
        # Initialize tools
        self.tools = [RAGSearchTool(config)]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        self.logger.info(f"RAG Agent initialized with {config.llm.provider} model: {config.llm.model_name}")
    
    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        llm_config = self.config.llm
        
        # Get API key from environment
        api_key = os.getenv(llm_config.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"{llm_config.api_key_env_var} environment variable is required for {llm_config.provider}. "
                f"Set it with: export {llm_config.api_key_env_var}='your-key-here'"
            )
        
        # Initialize based on provider
        if llm_config.provider.lower() == "openai":
            model_kwargs = {
                "model": llm_config.model_name,
                "temperature": llm_config.temperature,
                "api_key": api_key
            }
            
            # Add optional parameters
            if llm_config.max_tokens:
                model_kwargs["max_tokens"] = llm_config.max_tokens
            
            # Add any additional params from config
            model_kwargs.update(llm_config.params)
            
            return ChatOpenAI(**model_kwargs)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Create partial functions with bound parameters
        model_node = partial(call_model, model=self.model, tools=self.tools)
        tools_node = partial(call_tools, tools_by_name=self.tools_by_name)
        
        # Add nodes
        workflow.add_node("agent", model_node)
        workflow.add_node("tools", tools_node)
        
        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        # Compile the workflow
        return workflow.compile()
    
    def ask(self, question: str) -> str:
        """
        Ask the agent a question and get an answer using RAG.
        
        Args:
            question: The question to ask
            
        Returns:
            The agent's response
        """
        self.logger.info(f"Processing question: {question}")
        
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=question)]
            }
            
            # Run the workflow
            result = self.workflow.invoke(initial_state)
            
            # Extract the final response
            messages = result["messages"]
            final_message = messages[-1]
            
            if hasattr(final_message, 'content'):
                response = final_message.content
            else:
                response = str(final_message)
            
            self.logger.info("Question processed successfully")
            return response
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def chat(self, messages: List[BaseMessage]) -> str:
        """
        Have a conversation with the agent using a list of messages.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            The agent's response
        """
        try:
            # Create initial state with conversation history
            initial_state = {"messages": messages}
            
            # Run the workflow
            result = self.workflow.invoke(initial_state)
            
            # Extract the final response
            final_message = result["messages"][-1]
            
            if hasattr(final_message, 'content'):
                return final_message.content
            else:
                return str(final_message)
                
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get information about available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "args_schema": tool.args_schema.schema() if tool.args_schema else None
            }
            for tool in self.tools
        ]
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
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
                "directory": str(self.config.vectorstore.persist_directory),
                "collection": self.config.vectorstore.collection_name
            }
        } 
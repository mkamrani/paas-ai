"""
LangGraph nodes for RAG agent workflow.
"""

from typing import List, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from ..tools import RAGSearchTool


class AgentState(TypedDict):
    """State for the RAG agent."""
    messages: List[BaseMessage]


def call_model(state: AgentState, model: BaseChatModel, tools: List) -> AgentState:
    """
    Node that calls the language model with available tools.
    """
    messages = state["messages"]
    
    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)
    
    # Call the model
    response = model_with_tools.invoke(messages)
    
    # Return updated state
    return {"messages": messages + [response]}


def call_tools(state: AgentState, tools_by_name: dict) -> AgentState:
    """
    Node that executes tool calls.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Execute tool calls
    tool_messages = []
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            if tool_name in tools_by_name:
                tool = tools_by_name[tool_name]
                try:
                    result = tool.invoke(tool_args)
                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id,
                            name=tool_name
                        )
                    )
                except Exception as e:
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error executing {tool_name}: {str(e)}",
                            tool_call_id=tool_id,
                            name=tool_name
                        )
                    )
    
    return {"messages": messages + tool_messages}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Router function to determine if we should continue to tools or end.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the conversation
    return "end" 
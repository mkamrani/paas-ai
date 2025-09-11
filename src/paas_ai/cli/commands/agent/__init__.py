"""
Agent CLI commands module.

This module provides commands for interacting with the RAG agent:
- ask: Ask the agent a single question
- chat: Start an interactive chat session
"""

import click
from .ask import ask_command
from .chat import chat_command


@click.group(name="agent")
def agent_group():
    """Agent commands for testing RAG integration."""
    pass


# Register commands
agent_group.add_command(ask_command, name="ask")
agent_group.add_command(chat_command, name="chat")


__all__ = ["agent_group"] 
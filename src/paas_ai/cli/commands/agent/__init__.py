"""
Agent CLI commands module.

This module provides commands for interacting with the RAG agent:
- chat: Start an interactive chat session
"""

import click

from .chat import chat_command


@click.group(name="agent")
def agent_group():
    """Agent commands for testing RAG integration."""
    pass


# Register commands
agent_group.add_command(chat_command, name="chat")


__all__ = ["agent_group"]

"""
RAG (Retrieval Augmented Generation) CLI commands.

Provides command-line interface for managing RAG knowledge bases,
including resource management, search functionality, and configuration.
"""

import click


@click.group()
def rag():
    """
    🧠 Manage RAG (Retrieval-Augmented Generation) system.

    Commands for managing knowledge base resources, synchronization,
    and search operations.
    """
    pass


from .reports import report

# Import and register subcommand groups and commands
from .resources import resources_group
from .search import search
from .status import status
from .sync import sync

__all__ = ["rag", "resources_group", "search", "status", "sync", "report"]

# Add subcommands to the main rag group
rag.add_command(resources_group)
rag.add_command(search)
rag.add_command(status)
rag.add_command(sync)
rag.add_command(report)

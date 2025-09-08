"""
RAG (Retrieval-Augmented Generation) CLI commands.

This module organizes RAG commands into logical groups across separate files:
- resources: Resource management (add, remove, list)
- sync: Synchronization operations
- status: System status and health
- search: Knowledge base search
- reports: Analytics and reporting
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


# Import and register subcommand groups
from .resources import resources
from .sync import sync
from .status import status
from .search import search
from .reports import report

# Add subcommands to the main rag group
rag.add_command(resources)
rag.add_command(sync)
rag.add_command(status)
rag.add_command(search)
rag.add_command(report) 
"""
Persistence management for multi-agent conversations.

Provides checkpointer factory for LangGraph state persistence.
"""

from typing import Optional

from paas_ai.utils.logging import get_logger

from ..config.schemas import PersistenceConfig

logger = get_logger("paas_ai.agents.persistence")


def create_checkpointer(config: PersistenceConfig):
    """
    Create appropriate checkpointer based on configuration.

    Args:
        config: Persistence configuration

    Returns:
        Checkpointer instance or None if persistence disabled
    """
    if not config.enabled:
        logger.info("Persistence disabled, conversations will not be saved")
        return None

    try:
        if config.checkpointer_type == "memory":
            from langgraph.checkpoint.memory import MemorySaver

            logger.info("Using in-memory checkpointer (conversations lost on restart)")
            return MemorySaver()

        elif config.checkpointer_type == "sqlite":
            import os
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            sqlite_path = config.sqlite_path or "conversation_state.db"
            # Expand user home directory if present
            sqlite_path = os.path.expanduser(sqlite_path)
            # Ensure directory exists
            os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
            logger.info(f"Using SQLite checkpointer: {sqlite_path}")
            # Create connection and SqliteSaver directly
            conn = sqlite3.connect(sqlite_path, check_same_thread=False)
            return SqliteSaver(conn)

        elif config.checkpointer_type == "postgres":
            from langgraph.checkpoint.postgres import PostgresSaver

            if not config.postgres_url:
                raise ValueError("postgres_url is required for postgres checkpointer")
            logger.info(f"Using PostgreSQL checkpointer: {config.postgres_url}")
            return PostgresSaver(config.postgres_url)

        else:
            raise ValueError(f"Unknown checkpointer type: {config.checkpointer_type}")

    except ImportError as e:
        logger.error(f"Failed to import checkpointer for {config.checkpointer_type}: {e}")
        logger.warning("Falling back to no persistence")
        return None
    except Exception as e:
        logger.error(f"Failed to create {config.checkpointer_type} checkpointer: {e}")
        logger.warning("Falling back to no persistence")
        return None


def generate_thread_id() -> str:
    """
    Generate a unique thread ID for a new conversation.

    Returns:
        UUID-based thread identifier
    """
    import time
    import uuid

    # Use timestamp + uuid for readability and uniqueness
    timestamp = int(time.time())
    short_uuid = str(uuid.uuid4())[:8]

    return f"chat_{timestamp}_{short_uuid}"

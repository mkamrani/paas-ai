"""
Agent ask command - Ask the RAG agent a single question.
"""

import click
from typing import Optional

from ....core.config import load_config, ConfigurationError
from ....core.agents import RAGAgent
from ....utils.logging import get_logger

logger = get_logger("paas_ai.cli.agent.ask")


@click.command()
@click.option("--question", "-q", required=True, help="Question to ask the agent")
@click.option("--config-profile", help="Override config profile for this operation")
@click.option("--show-config", is_flag=True, help="Show configuration summary")
def ask_command(question: str, config_profile: Optional[str], show_config: bool):
    """Ask the RAG agent a question."""
    try:
        # Load configuration
        if config_profile:
            logger.warning(f"Config profile override not yet implemented: {config_profile}")
        
        config = load_config()
        logger.info(f"Using configuration with {config.embedding.type} embeddings")
        
        # Initialize agent
        agent = RAGAgent(config)
        
        # Show config summary if requested
        if show_config:
            config_summary = agent.get_config_summary()
            click.echo("\n" + "="*60)
            click.echo("CONFIGURATION SUMMARY:")
            click.echo("="*60)
            click.echo(f"LLM: {config_summary['llm']['provider']} ({config_summary['llm']['model']})")
            click.echo(f"Embedding: {config_summary['embedding']['type']} ({config_summary['embedding']['model']})")
            click.echo(f"VectorStore: {config_summary['vectorstore']['type']} -> {config_summary['vectorstore']['directory']}")
            click.echo(f"Collection: {config_summary['vectorstore']['collection']}")
            click.echo("="*60 + "\n")
        
        # Ask the question
        logger.info(f"Asking: {question}")
        response = agent.ask(question)
        
        # Display response
        click.echo("\n" + "="*60)
        click.echo("AGENT RESPONSE:")
        click.echo("="*60)
        click.echo(response)
        click.echo("="*60 + "\n")
        
        return True
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Try creating a config file with: paas-ai config init")
        click.echo(click.style(f"❌ Configuration error: {e}", fg="red"))
        click.echo(click.style("💡 Try creating a config file with: paas-ai config init", fg="yellow"))
        return False
    except Exception as e:
        logger.error(f"Failed to process question: {e}")
        click.echo(click.style(f"❌ Failed to process question: {e}", fg="red"))
        return False 
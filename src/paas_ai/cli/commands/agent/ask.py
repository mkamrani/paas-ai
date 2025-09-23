"""
Agent ask command - Ask the RAG agent a single question.
"""

from typing import Optional

import click

from paas_ai.core.agents import RAGAgent
from paas_ai.core.config import ConfigurationError, load_config
from paas_ai.utils.logging import get_logger

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
            # Use the config profiles system like RAG commands
            from paas_ai.core.config.schemas import DEFAULT_CONFIG_PROFILES

            if config_profile in DEFAULT_CONFIG_PROFILES:
                config = DEFAULT_CONFIG_PROFILES[config_profile]
                logger.info(f"Using config profile: {config_profile}")
            else:
                logger.warning(f"Unknown config profile '{config_profile}', using default")
                config = load_config()
        else:
            config = load_config()

        logger.info(f"Using configuration with {config.embedding.type} embeddings")

        # Initialize agent
        agent = RAGAgent(config)

        # Show config summary if requested
        if show_config:
            try:
                config_summary = agent.get_config_summary()
                click.echo("\n" + "=" * 60)
                click.echo("CONFIGURATION SUMMARY:")
                click.echo("=" * 60)
                click.echo(
                    f"LLM: {config_summary['llm']['provider']} ({config_summary['llm']['model']})"
                )
                click.echo(
                    f"Embedding: {config_summary['embedding']['type']} ({config_summary['embedding']['model']})"
                )
                click.echo(
                    f"VectorStore: {config_summary['vectorstore']['type']} -> {config_summary['vectorstore']['directory']}"
                )
                click.echo(f"Collection: {config_summary['vectorstore']['collection']}")

                # Show multi-agent info if available
                if "multi_agent" in config_summary:
                    ma_config = config_summary["multi_agent"]
                    click.echo(f"Multi-Agent Mode: {ma_config['mode']}")
                    click.echo(f"Agents: {', '.join(ma_config['agents'])}")
                    click.echo(f"Token Tracking: {'ON' if ma_config['track_tokens'] else 'OFF'}")
                    click.echo(f"Verbose Mode: {'ON' if ma_config['verbose'] else 'OFF'}")

                click.echo("=" * 60 + "\n")
            except Exception as e:
                logger.warning(f"Failed to show config summary: {e}")
                click.echo(
                    click.style(f"⚠️  Warning: Could not display config summary: {e}", fg="yellow")
                )
                click.echo("")

        # Ask the question
        logger.info(f"Asking: {question}")
        response = agent.ask(question)

        # Display response
        click.echo("\n" + "=" * 60)
        click.echo("AGENT RESPONSE:")
        click.echo("=" * 60)
        click.echo(response)
        click.echo("=" * 60 + "\n")

        return True

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Try creating a config file with: paas-ai config init")
        click.echo(click.style(f"❌ Configuration error: {e}", fg="red"))
        click.echo(
            click.style("💡 Try creating a config file with: paas-ai config init", fg="yellow")
        )
        return False
    except Exception as e:
        logger.error(f"Failed to process question: {e}")
        click.echo(click.style(f"❌ Failed to process question: {e}", fg="red"))
        return False

"""
RAG Search commands.

Search the knowledge base for relevant information.
"""

import json

import click

from paas_ai.core.config import ResourceType, load_config
from paas_ai.core.rag import ConfigurationError, RAGProcessor
from paas_ai.utils.logging import get_logger


@click.command()
@click.argument("query")
@click.option(
    "--type",
    "resource_type",
    type=click.Choice(["dsl", "contextual", "guidelines", "domain_rules", "all"]),
    default="all",
    help="Search specific resource type",
)
@click.option("--limit", type=int, default=5, help="Maximum number of results")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.option(
    "--config-profile",
    type=click.Choice(["default", "local", "production", "citation_demo"]),
    default="default",
    help="Configuration profile to use",
)
def search(query: str, resource_type: str, limit: int, format: str, config_profile: str):
    """
    🔍 Search the knowledge base.

    Examples:
      paas-ai rag search "kubernetes deployment"
      paas-ai rag search "security guidelines" --type guidelines
      paas-ai rag search "microservice patterns" --limit 10 --format json
    """
    logger = get_logger("paas_ai.cli.rag")
    logger.set_context("RAG-SEARCH")

    try:
        logger.info(f"Searching for: '{query}'")
        logger.debug(f"Resource type: {resource_type}, Limit: {limit}")

        # Initialize RAG processor
        if config_profile != "default":
            # TODO: Implement profile override logic or remove the option
            logger.warning(f"Config profile override not yet implemented: {config_profile}")

        config = load_config()
        processor = RAGProcessor(config)

        # Check if knowledge base is available
        stats = processor.get_stats()
        if stats["status"] == "empty":
            logger.warning(
                "Knowledge base is empty. Add resources first using 'paas-ai rag resources add'"
            )
            return

        logger.progress("Executing search...")

        # Convert resource type filter
        resource_filter = None
        if resource_type != "all":
            resource_filter = ResourceType(resource_type)

        # Perform search
        results = processor.search(
            query=query, resource_type=resource_filter, limit=limit, include_metadata=True
        )

        if not results:
            logger.warning("No results found")
            return

        logger.progress("Ranking results...")

        if format == "text":
            logger.info(f"\n🔍 Search Results ({len(results)} found):")
            logger.info("-" * 50)

            for i, result in enumerate(results, 1):
                logger.info(f"{i}. Score: {result.get('score', 0.0):.3f}")
                logger.info(f"   Source: {result['metadata']['source_url']}")
                logger.info(f"   Type: {result['metadata']['resource_type']}")

                if result["metadata"].get("tags"):
                    logger.info(f"   Tags: {', '.join(result['metadata']['tags'])}")

                # Show content preview (first 200 chars)
                content_preview = result["content"][:200]
                if len(result["content"]) > 200:
                    content_preview += "..."

                logger.info(f"   Content: {content_preview}")
                logger.info("")

        else:
            # JSON format
            output = {"query": query, "total_results": len(results), "results": results}
            click.echo(json.dumps(output, indent=2))

        logger.success(f"Search completed - {len(results)} results found")

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise click.ClickException(str(e))
    except ValueError as e:
        logger.error(str(e))
        raise click.ClickException(str(e))
    except Exception as e:
        logger.exception(f"Search failed: {e}")
        raise click.ClickException(f"Search failed: {e}")
    finally:
        logger.clear_context()

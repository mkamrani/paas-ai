"""
RAG resources management commands.

Provides CLI commands for adding, listing, and managing RAG knowledge base resources.
"""

import asyncio
import csv
from pathlib import Path
from typing import List, Optional

import click

from paas_ai.core.config import ConfigurationError, ResourceType, load_config
from paas_ai.core.rag import RAGProcessor, create_resource_from_url
from paas_ai.utils.logging import get_logger

logger = get_logger("paas_ai.cli.rag.resources")


@click.group(name="resources")
def resources_group():
    """Manage RAG knowledge base resources."""
    pass


@resources_group.command("add")
@click.option("--url", required=True, help="URL or file path to add to knowledge base")
@click.option(
    "--type",
    "resource_type",
    type=click.Choice([t.value for t in ResourceType]),
    default=ResourceType.DSL.value,
    help="Type of resource content",
)
@click.option("--tags", default="", help="Comma-separated tags for the resource")
@click.option("--priority", type=int, default=5, help="Priority level (1-10)")
@click.option("--config-profile", help="Override config profile for this operation")
def add_resource(
    url: str, resource_type: str, tags: str, priority: int, config_profile: Optional[str]
):
    """Add a single resource to the knowledge base."""
    try:
        # Load configuration with fallback hierarchy
        if config_profile:
            # TODO: Implement profile override logic or get rid of the option
            logger.warning(f"Config profile override not yet implemented: {config_profile}")

        config = load_config()
        logger.info(f"Using configuration profile with {config.embedding.type} embeddings")

        # Create resource
        resource_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        resource = create_resource_from_url(
            url=url,
            resource_type=ResourceType(resource_type),
            tags=resource_tags,
            priority=priority,
        )

        # Process resource
        processor = RAGProcessor(config)
        results = asyncio.run(processor.add_resources([resource]))

        # Report results
        if results["successful"] > 0:
            logger.success(f"Successfully processed {results['successful']} resources")
            logger.info(f"Total documents: {results['total_documents']}")
        else:
            logger.error("Failed to process any resources")
            raise click.ClickException("Failed to process any resources")

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Try creating a config file with: paas-ai config init")
        raise click.ClickException(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"Failed to add resource: {e}")
        raise click.ClickException(f"Failed to add resource: {e}")


@resources_group.command("add-batch")
@click.option(
    "--csv-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="CSV file with resources to add",
)
@click.option("--config-profile", help="Override config profile for this operation")
def add_resources_batch(csv_file: Path, config_profile: Optional[str]):
    """Add multiple resources from a CSV file."""
    try:
        # Load configuration
        if config_profile:
            logger.warning(f"Config profile override not yet implemented: {config_profile}")

        config = load_config()
        logger.info(f"Using configuration profile with {config.embedding.type} embeddings")

        # Parse CSV file
        resources = []
        with open(csv_file, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                url = row.get("url")
                resource_type = row.get("type", "dsl")
                tags = row.get("tags", "").split(",") if row.get("tags") else []
                priority = int(row.get("priority", 5))

                if not url:
                    continue

                resource = create_resource_from_url(
                    url=url,
                    resource_type=ResourceType(resource_type),
                    tags=[tag.strip() for tag in tags if tag.strip()],
                    priority=priority,
                )
                resources.append(resource)

        if not resources:
            logger.error("No valid resources found in CSV file")
            raise click.ClickException("No valid resources found in CSV file")

        logger.info(f"Found {len(resources)} resources in CSV file")

        # Process resources
        processor = RAGProcessor(config)
        results = asyncio.run(processor.add_resources(resources))

        # Report results
        logger.success(f"Batch processing completed:")
        logger.info(f"  Successful: {results['successful']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info(f"  Total documents: {results['total_documents']}")

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Try creating a config file with: paas-ai config init")
        raise click.ClickException(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"Failed to add resources: {e}")
        raise click.ClickException(f"Failed to add resources: {e}")


@resources_group.command("list")
def list_resources():
    """List all resources in the knowledge base."""
    # TODO: Implement resource listing
    logger.info("Resource listing not yet implemented")


@resources_group.command("remove")
@click.argument("resource_id")
def remove_resource(resource_id: str):
    """Remove a resource from the knowledge base."""
    # TODO: Implement resource removal
    logger.info(f"Resource removal not yet implemented: {resource_id}")


@resources_group.command("clear")
@click.confirmation_option(prompt="Are you sure you want to clear all resources?")
def clear_resources():
    """Clear all resources from the knowledge base."""
    try:
        config = load_config()
        processor = RAGProcessor(config)
        processor.clear_knowledge_base()
        logger.success("Knowledge base cleared successfully")
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise click.ClickException(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"Failed to clear knowledge base: {e}")
        raise click.ClickException(f"Failed to clear knowledge base: {e}")


# For backward compatibility, keep the old function name
add_resources = add_resource

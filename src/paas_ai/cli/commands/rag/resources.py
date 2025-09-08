"""
RAG Resource management commands.

Handle adding, removing, and listing knowledge base resources.
"""

import click
import csv
import json
from pathlib import Path
from typing import Optional, List

from ....utils.logging import get_logger
from ....core.rag import RAGProcessor, ResourceConfig, create_resource_from_url, ConfigurationError
from ....core.rag.config import DEFAULT_CONFIGS, ResourceType


@click.group()
def resources():
    """
    📚 Manage knowledge base resources.
    
    Add, remove, and list resources in the RAG knowledge base.
    """
    pass


@resources.command()
@click.option(
    '--type', 'resource_type',
    type=click.Choice(['dsl', 'contextual', 'guidelines', 'domain_rules', 'all']),
    default='all',
    help='Show specific resource type'
)
@click.option(
    '--status',
    type=click.Choice(['active', 'stale', 'error']),
    help='Filter by status'
)
@click.option(
    '--format',
    type=click.Choice(['table', 'json']),
    default='table',
    help='Output format'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed information'
)
def list_resources(resource_type: str, status: Optional[str], format: str, verbose: bool):
    """
    📋 List knowledge base resources.
    
    Examples:
      paas-ai rag resources list
      paas-ai rag resources list --type dsl --format json
      paas-ai rag resources list --status stale --verbose
    """
    logger = get_logger("paas_ai.cli.rag")
    logger.set_context("RAG-LIST")
    
    try:
        # Initialize RAG processor
        config = DEFAULT_CONFIGS['default']  # TODO: Load from config file
        processor = RAGProcessor(config)
        
        # Get stats
        stats = processor.get_stats()
        
        if format == "table":
            logger.info("📚 Knowledge Base Resources")
            logger.info("=" * 50)
            logger.info(f"Total documents: {stats['total_documents']}")
            logger.info(f"Vector store: {stats['vectorstore_type']}")
            logger.info(f"Embedding model: {stats['embedding_model']}")
            logger.info(f"Status: {stats['status']}")
            
            if verbose:
                logger.info(f"\nRetriever type: {stats.get('retriever_type', 'N/A')}")
                # TODO: Add more detailed stats when we have resource tracking
                
        else:
            # JSON format
            logger.info(json.dumps(stats, indent=2))
        
        logger.success("Resource listing completed")
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise click.ClickException(str(e))
    except Exception as e:
        logger.exception(f"Failed to list resources: {e}")
        raise click.ClickException(f"List operation failed: {e}")
    finally:
        logger.clear_context()


@resources.command()
@click.option(
    '--csv-file',
    type=click.Path(exists=True, path_type=Path),
    help='CSV file containing resource definitions'
)
@click.option(
    '--url',
    help='Single URL to add'
)
@click.option(
    '--type', 'resource_type',
    type=click.Choice(['dsl', 'contextual', 'guidelines', 'domain_rules']),
    help='Resource type (required for single URL)'
)
@click.option(
    '--config-profile',
    type=click.Choice(['default', 'local', 'production']),
    default='default',
    help='Configuration profile to use'
)
@click.option(
    '--priority',
    type=click.IntRange(1, 10),
    default=1,
    help='Resource priority (1-10)'
)
@click.option(
    '--tags',
    help='Comma-separated tags for the resource'
)
@click.option(
    '--chunk-size',
    type=int,
    help='Custom chunk size for text splitting'
)
@click.option(
    '--chunk-overlap',
    type=int,
    help='Custom chunk overlap for text splitting'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be added without making changes'
)
def add_resources(
    csv_file: Optional[Path],
    url: Optional[str],
    resource_type: Optional[str],
    config_profile: str,
    priority: int,
    tags: Optional[str],
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    dry_run: bool,
):
    """
    ➕ Add new resources to the knowledge base.
    
    Examples:
      paas-ai rag resources add --csv-file new_sources.csv
      paas-ai rag resources add --url "https://k8s.io/docs" --type dsl
      paas-ai rag resources add --url "https://example.com" --type guidelines --chunk-size 1500
    """
    logger = get_logger("paas_ai.cli.rag")
    logger.set_context("RAG-ADD")
    
    try:
        if not csv_file and not url:
            raise click.ClickException("Either --csv-file or --url must be specified")
        
        if url and not resource_type:
            raise click.ClickException("--type is required when using --url")
        
        if dry_run:
            logger.warning("DRY RUN MODE - No resources will be added")
        
        # Initialize RAG processor
        config = DEFAULT_CONFIGS[config_profile]
        processor = RAGProcessor(config)
        
        resources = []
        
        if csv_file:
            logger.info(f"Loading resources from CSV file: {csv_file}")
            resources.extend(_load_resources_from_csv(csv_file))
            
        if url:
            logger.info(f"Adding single resource: {url}")
            
            # Parse tags
            tag_list = []
            if tags:
                tag_list = [tag.strip() for tag in tags.split(',')]
            
            # Create resource config with custom settings
            kwargs = {
                'priority': priority,
                'tags': tag_list,
            }
            
            if chunk_size:
                kwargs['chunk_size'] = chunk_size
            if chunk_overlap:
                kwargs['chunk_overlap'] = chunk_overlap
                
            resource = create_resource_from_url(
                url, 
                ResourceType(resource_type),
                **kwargs
            )
            resources.append(resource)
        
        if not resources:
            logger.warning("No resources to add")
            return
        
        logger.info(f"Processing {len(resources)} resources")
        
        if dry_run:
            for i, resource in enumerate(resources, 1):
                logger.info(f"{i}. {resource.url} ({resource.resource_type.value})")
                logger.info(f"   Loader: {resource.loader.type}")
                logger.info(f"   Splitter: {resource.splitter.type}")
                logger.info(f"   Chunk size: {resource.splitter.chunk_size}")
                logger.info(f"   Priority: {resource.priority}")
                logger.info(f"   Tags: {resource.tags}")
            return
        
        # Process resources
        results = processor.add_resources(resources)
        
        # Show results
        logger.success(f"Successfully processed {results['successful']} resources")
        logger.info(f"Total documents added: {results['total_documents']}")
        
        if results['failed'] > 0:
            logger.warning(f"Failed to process {results['failed']} resources")
            for error in results['errors']:
                logger.error(f"  - {error}")
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise click.ClickException(str(e))
    except Exception as e:
        logger.exception(f"Failed to add resources: {e}")
        raise click.ClickException(f"Add operation failed: {e}")
    finally:
        logger.clear_context()


@resources.command()
@click.option(
    '--all',
    'clear_all',
    is_flag=True,
    help='Remove all resources (requires confirmation)'
)
@click.option(
    '--url-pattern',
    help='Remove resources matching URL pattern'
)
@click.option(
    '--type', 'resource_type',
    type=click.Choice(['dsl', 'contextual', 'guidelines', 'domain_rules']),
    help='Remove resources of specific type'
)
@click.option(
    '--force',
    is_flag=True,
    help='Skip confirmation prompts'
)
def remove(
    clear_all: bool,
    url_pattern: Optional[str],
    resource_type: Optional[str],
    force: bool
):
    """
    🗑️ Remove resources from the knowledge base.
    
    Examples:
      paas-ai rag resources remove --all --force
      paas-ai rag resources remove --type dsl
      paas-ai rag resources remove --url-pattern "*.example.com"
    """
    logger = get_logger("paas_ai.cli.rag")
    logger.set_context("RAG-REMOVE")
    
    try:
        if not any([clear_all, url_pattern, resource_type]):
            raise click.ClickException("Must specify --all, --url-pattern, or --type")
        
        # Initialize RAG processor
        config = DEFAULT_CONFIGS['default']
        processor = RAGProcessor(config)
        
        if clear_all:
            if not force:
                if not click.confirm("⚠️ This will remove ALL resources. Are you sure?"):
                    logger.info("Operation cancelled")
                    return
            
            processor.clear_knowledge_base()
            logger.success("All resources removed")
        else:
            # TODO: Implement selective removal when we have resource tracking
            logger.warning("Selective removal not yet implemented. Use --all to clear everything.")
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise click.ClickException(str(e))
    except Exception as e:
        logger.exception(f"Failed to remove resources: {e}")
        raise click.ClickException(f"Remove operation failed: {e}")
    finally:
        logger.clear_context()


def _load_resources_from_csv(csv_file: Path) -> List[ResourceConfig]:
    """Load resources from a CSV file."""
    resources = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            url = row.get('url', '').strip()
            resource_type = row.get('type', '').strip()
            
            if not url or not resource_type:
                continue
            
            # Parse optional fields
            priority = int(row.get('priority', 1))
            tags = [tag.strip() for tag in row.get('tags', '').split(',') if tag.strip()]
            chunk_size = int(row.get('chunk_size')) if row.get('chunk_size') else None
            chunk_overlap = int(row.get('chunk_overlap')) if row.get('chunk_overlap') else None
            
            # Create resource config
            kwargs = {
                'priority': priority,
                'tags': tags,
            }
            
            if chunk_size:
                kwargs['chunk_size'] = chunk_size
            if chunk_overlap:
                kwargs['chunk_overlap'] = chunk_overlap
            
            resource = create_resource_from_url(
                url,
                ResourceType(resource_type),
                **kwargs
            )
            resources.append(resource)
    
    return resources 
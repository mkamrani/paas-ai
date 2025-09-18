"""
RAG Status and health monitoring commands.

Show system status and resource health information.
"""

import click
import json

from paas_ai.utils.logging import get_logger
from paas_ai.core.rag import RAGProcessor, ConfigurationError
from paas_ai.core.config import load_config, DEFAULT_CONFIG_PROFILES


@click.command()
@click.option(
    '--detailed',
    is_flag=True,
    help='Show detailed status information'
)
@click.option(
    '--format',
    type=click.Choice(['text', 'json']),
    default='text',
    help='Output format'
)
@click.option(
    '--config-profile',
    type=click.Choice(['default', 'local', 'production']),
    help='Configuration profile to use (overrides current profile)'
)
def status(detailed: bool, format: str, config_profile: str):
    """
    📊 Show RAG system status and resource health.
    
    Examples:
      paas-ai rag status
      paas-ai rag status --detailed
      paas-ai rag status --format json
    """
    logger = get_logger("paas_ai.cli.rag")
    logger.set_context("RAG-STATUS")
    
    try:
        # Load configuration using the profile system
        if config_profile:
            # Use specific built-in profile if provided
            config = DEFAULT_CONFIG_PROFILES[config_profile]
        else:
            # Use current active profile from config file/.env
            config = load_config()
        
        # Initialize RAG processor
        processor = RAGProcessor(config)
        
        # Get stats
        stats = processor.get_stats()
        
        if format == 'text':
            logger.info("RAG System Status")
            logger.info("=" * 50)
            
            # Basic status
            logger.info(f"📚 Knowledge Base Status: {stats['status']}")
            logger.info(f"📊 Total Documents: {stats['total_documents']}")
            logger.info(f"🔧 Vector Store: {stats['vectorstore_type']}")
            logger.info(f"🤖 Embedding Model: {stats['embedding_model']}")
            
            if stats['status'] != 'empty':
                logger.info(f"🔍 Retriever Type: {stats.get('retriever_type', 'N/A')}")
            
            if detailed:
                logger.info("\n🔍 Detailed Configuration:")
                logger.info(f"  - Profile: {config_profile}")
                logger.info(f"  - Batch Size: {config.batch_size}")
                logger.info(f"  - Max Retries: {config.max_retries}")
                logger.info(f"  - Timeout: {config.timeout}s")
                logger.info(f"  - URL Validation: {config.validate_urls}")
                logger.info(f"  - Skip Invalid Docs: {config.skip_invalid_docs}")
                
                if config.vectorstore.persist_directory:
                    persist_path = config.vectorstore.persist_directory
                    exists = persist_path.exists() if persist_path else False
                    logger.info(f"  - Persist Directory: {persist_path}")
                    logger.info(f"  - Directory Exists: {exists}")
                    
                logger.info(f"\n⚙️ Embedding Configuration:")
                logger.info(f"  - Type: {config.embedding.type}")
                logger.info(f"  - Model: {config.embedding.model_name}")
                
                logger.info(f"\n🗄️ Vector Store Configuration:")
                logger.info(f"  - Type: {config.vectorstore.type}")
                logger.info(f"  - Collection: {config.vectorstore.collection_name}")
                
                logger.info(f"\n🔍 Retriever Configuration:")
                logger.info(f"  - Type: {config.retriever.type}")
                logger.info(f"  - Search Kwargs: {config.retriever.search_kwargs}")
            
            # Health indicators
            if stats['status'] == 'empty':
                logger.warning("\n⚠️ Knowledge base is empty")
                logger.info("💡 Add resources using: paas-ai rag resources add")
            else:
                logger.success("\n✅ RAG system operational")
        
        else:
            # JSON format
            detailed_stats = {
                **stats,
                'config_profile': config_profile,
                'configuration': {
                    'batch_size': config.batch_size,
                    'max_retries': config.max_retries,
                    'timeout': config.timeout,
                    'validate_urls': config.validate_urls,
                    'skip_invalid_docs': config.skip_invalid_docs,
                    'embedding': {
                        'type': config.embedding.type,
                        'model_name': config.embedding.model_name,
                    },
                    'vectorstore': {
                        'type': config.vectorstore.type,
                        'collection_name': config.vectorstore.collection_name,
                        'persist_directory': str(config.vectorstore.persist_directory) if config.vectorstore.persist_directory else None,
                    },
                    'retriever': {
                        'type': config.retriever.type,
                        'search_kwargs': config.retriever.search_kwargs,
                    }
                } if detailed else {}
            }
            logger.info(json.dumps(detailed_stats, indent=2))
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise click.ClickException(str(e))
    except Exception as e:
        logger.exception(f"Failed to get status: {e}")
        raise click.ClickException(f"Status check failed: {e}")
    finally:
        logger.clear_context() 
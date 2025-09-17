"""
RAG Pipeline processor.

Main orchestrator for the RAG pipeline that coordinates document loading,
processing, embedding, and storage following LangChain patterns.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio
import logging
from urllib.parse import urlparse
import requests
from pathlib import Path
import time

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from .config import (
    Config, ResourceConfig, ResourceType,
    get_default_loader_config, get_default_splitter_config
)
from .loaders import DocumentLoaderFactory
from .splitters import TextSplitterFactory
from .embeddings import EmbeddingsFactory
from .vectorstore import VectorStoreFactory
from .retrievers import RetrieverFactory
from .processing import (
    ProcessingPipeline, ProcessingStage, ProcessingContext
)
from paas_ai.utils.logging import get_logger


def convert_main_config_to_rag_config(main_config) -> Config:
    """Convert main config to RAG config format."""
    # Import here to avoid circular imports
    try:
        from ..config.schemas import Config as MainConfig
        if isinstance(main_config, MainConfig):
            # Import RAG config classes
            from .config import (
                EmbeddingConfig as RagEmbeddingConfig,
                VectorStoreConfig as RagVectorStoreConfig,
                RetrieverConfig as RagRetrieverConfig
            )
            
            # Convert config objects to RAG format by extracting dict values
            rag_config = Config(
                embedding=RagEmbeddingConfig(
                    type=main_config.embedding.type,
                    model_name=main_config.embedding.model_name,
                    params=main_config.embedding.params
                ),
                vectorstore=RagVectorStoreConfig(
                    type=main_config.vectorstore.type,
                    persist_directory=main_config.vectorstore.persist_directory,
                    collection_name=main_config.vectorstore.collection_name,
                    params=main_config.vectorstore.params
                ),
                retriever=RagRetrieverConfig(
                    type=main_config.retriever.type,
                    search_kwargs=main_config.retriever.search_kwargs,
                    params=main_config.retriever.params
                ),
                batch_size=getattr(main_config, 'batch_size', 32),
                validate_urls=getattr(main_config, 'validate_urls', True),
                log_level=getattr(main_config, 'log_level', 'INFO')
            )
            
            # Add citation config if available
            if hasattr(main_config, 'citation') and main_config.citation:
                rag_config.citation = main_config.citation
            
            return rag_config
    except ImportError:
        pass
    
    # If already RAG config or conversion failed, return as-is
    return main_config


class ValidationError(Exception):
    """Raised when resource validation fails."""
    pass


class ProcessingError(Exception):
    """Raised when document processing fails."""
    pass


class ConfigurationError(Exception):
    """Raised when RAG configuration is invalid or missing required settings."""
    pass


class VectorStoreStage(ProcessingStage):
    """Pipeline stage for storing documents in vectorstore."""
    
    def __init__(self, rag_processor: 'RAGProcessor'):
        super().__init__("vectorstore")
        self.rag_processor = rag_processor
    
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """Store documents in vectorstore."""
        if not context.documents:
            return context
        
        # Filter complex metadata for compatibility with vectorstores like Chroma
        from langchain_community.vectorstores.utils import filter_complex_metadata
        filtered_documents = filter_complex_metadata(context.documents)
        
        if not self.rag_processor.vectorstore:
            # Create new vectorstore
            self.rag_processor.vectorstore = VectorStoreFactory.create_vectorstore(
                self.rag_processor.config.vectorstore,
                self.rag_processor.embeddings,
                filtered_documents
            )
            self.rag_processor.logger.info("Created new vectorstore")
        else:
            # Add to existing vectorstore
            self.rag_processor.vectorstore.add_documents(filtered_documents)
            self.rag_processor.logger.info(f"Added {len(filtered_documents)} documents to existing vectorstore")
        
        # Create/update retriever
        self.rag_processor.retriever = RetrieverFactory.create_retriever(
            self.rag_processor.config.retriever,
            self.rag_processor.vectorstore
        )
        
        return context


class RAGProcessor:
    """Main RAG pipeline processor."""
    
    def __init__(self, config: Union[Config, Any]):
        """Initialize the RAG processor with configuration."""
        # Convert main config to RAG config if needed
        self.config = convert_main_config_to_rag_config(config)
        self.logger = get_logger("paas_ai.rag.pipeline")
        
        # Initialize components with proper error handling
        try:
            self.embeddings = EmbeddingsFactory.create_embeddings(config.embedding)
        except Exception as e:
            self._handle_initialization_error(e, config)
            
        self.vectorstore = None
        self.retriever = None
        
        # Initialize citation enricher if enabled (use converted config)
        self.citation_enricher = None
        if hasattr(self.config, 'citation') and self.config.citation and self.config.citation.enabled:
            try:
                from .citations import CitationEnricher
                self.citation_enricher = CitationEnricher(self.config.citation)
                self.logger.info(f"Citation system enabled with verbosity: {self.config.citation.verbosity}")
            except ImportError:
                self.logger.warning("Citation system requested but citations module not available")
        elif hasattr(self.config, 'citation') and self.config.citation:
            self.logger.debug("Citation system available but not enabled")
        
        # Load existing vectorstore if available
        self._load_existing_vectorstore()
    
    def _create_processing_pipeline(self) -> ProcessingPipeline:
        """Create the document processing pipeline."""
        from .processing.stages import LoadStage, ValidateStage, SplitStage, EnrichStage
        
        # Create pipeline with standard stages + vectorstore stage
        pipeline = (LoadStage() | 
                   ValidateStage() | 
                   SplitStage() | 
                   EnrichStage() |
                   VectorStoreStage(self))
        
        return pipeline
    
    def _handle_initialization_error(self, error: Exception, config: Any):
        """Handle initialization errors with more specific error messages."""
        error_str = str(error).lower()
        
        # Torch meta tensor issues (common with newer PyTorch and SentenceTransformers)
        if "meta tensor" in error_str or "cannot copy out of meta tensor" in error_str:
            raise ConfigurationError(
                f"PyTorch meta tensor error detected with {config.embedding.type} embeddings.\n"
                f"This is commonly caused by:\n"
                f"  1. Compatibility issues between PyTorch and sentence-transformers versions\n"
                f"  2. Using deprecated langchain-community HuggingFaceEmbeddings\n"
                f"  3. Insufficient GPU memory during model loading\n"
                f"Solutions:\n"
                f"  1. Install langchain-huggingface: pip install langchain-huggingface\n"
                f"  2. Use a different embedding model: --config-profile default (OpenAI)\n"
                f"  3. Downgrade PyTorch: pip install torch<2.8.0\n"
                f"Original error: {error}"
            ) from error
        
        # OpenAI API key missing
        elif "openai" in error_str and ("api" in error_str or "key" in error_str):
            raise ConfigurationError(
                f"OpenAI API key is required for OpenAI embeddings.\n"
                f"Set environment variable: export OPENAI_API_KEY='your-key-here'\n"
                f"Or use local config: --config-profile local"
            ) from error
        
        # Cohere API key missing
        elif "cohere" in error_str and "api" in error_str:
            raise ConfigurationError(
                f"Cohere API key is required for Cohere embeddings.\n"
                f"Set environment variable: export COHERE_API_KEY='your-key-here'\n"
                f"Or use local config: --config-profile local"
            ) from error
        
        # HuggingFace/SentenceTransformers model download issues
        elif any(term in error_str for term in ["huggingface", "sentence", "transformers", "download"]):
            raise ConfigurationError(
                f"Failed to load model '{config.embedding.model_name}'.\n"
                f"This might be due to:\n"
                f"  1. Network connectivity issues\n"
                f"  2. Invalid model name\n"
                f"  3. Missing dependencies\n"
                f"Original error: {error}"
            ) from error
        
        # Generic fallback
        else:
            raise ConfigurationError(
                f"Failed to initialize RAG system with {config.embedding.type} embeddings.\n"
                f"Config profile: Check your configuration settings.\n"
                f"Original error: {error}"
            ) from error
    
    def _load_existing_vectorstore(self):
        """Load existing vectorstore from disk if available."""
        try:
            self.logger.debug(f"Attempting to load vectorstore from {self.config.vectorstore.persist_directory}")
            self.vectorstore = VectorStoreFactory.load_vectorstore(
                self.config.vectorstore,
                self.embeddings
            )
            if self.vectorstore:
                self.logger.info("Loaded existing vectorstore")
                self.retriever = RetrieverFactory.create_retriever(
                    self.config.retriever,
                    self.vectorstore
                )
            else:
                self.logger.debug("No existing vectorstore found")
        except Exception as e:
            self.logger.warning(f"Failed to load existing vectorstore: {e}")
            self.logger.debug(f"Exception details: {e}", exc_info=True)
    
    def validate_resource(self, resource: ResourceConfig) -> None:
        """Validate a resource configuration."""
        if not self.config.validate_urls:
            return
        
        url = resource.url
        self.logger.debug(f"Validating resource: {url}")
        
        # Parse URL
        parsed = urlparse(url)
        
        # Check if it's a local file/directory
        if not parsed.scheme:
            path = Path(url)
            if not path.exists():
                raise ValidationError(f"Local path does not exist: {url}")
            return
        
        # Check if it's a valid web URL
        if parsed.scheme in ['http', 'https']:
            try:
                response = requests.head(url, timeout=10, allow_redirects=True)
                if response.status_code >= 400:
                    raise ValidationError(f"URL returned status {response.status_code}: {url}")
            except requests.RequestException as e:
                raise ValidationError(f"Failed to access URL {url}: {e}")
            return
        
        # Check special URL schemes
        if parsed.scheme in ['confluence', 'notion', 'github']:
            # These require special validation, but we'll skip for now
            return
        
        raise ValidationError(f"Unsupported URL scheme: {parsed.scheme}")
    
    async def add_resources(self, resources: List[ResourceConfig]) -> Dict[str, Any]:
        """Add multiple resources to the knowledge base using the processing pipeline."""
        self.logger.info(f"Adding {len(resources)} resources to knowledge base")
        
        # Create the processing pipeline
        pipeline = self._create_processing_pipeline()
        
        # Process resources with citation enricher injection
        results = []
        for resource in resources:
            # Create context and inject citation enricher if available
            context = ProcessingContext(resource=resource)
            if self.citation_enricher:
                context.citation_enricher = self.citation_enricher
            
            # Process single resource
            result = await pipeline.process_with_context(context)
            results.append(result)
        
        # Aggregate results
        successful = sum(1 for r in results if r.success)
        total_docs = sum(len(r.context.documents) for r in results if r.success)
        errors = [r.error for r in results if not r.success and r.error]
        
        self.logger.success(
            f"Completed processing. "
            f"Successful: {successful}, "
            f"Failed: {len(resources) - successful}, "
            f"Total documents: {total_docs}"
        )
        
        return {
            'total_resources': len(resources),
            'successful': successful,
            'failed': len(resources) - successful,
            'total_documents': total_docs,
            'errors': errors
        }
    
    def search(
        self,
        query: str,
        resource_type: Optional[ResourceType] = None,
        limit: int = 5,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base."""
        if not self.retriever:
            raise ValueError("No retriever available. Add resources first.")
        
        self.logger.debug(f"Searching for: '{query}'")
        
        # Update search kwargs with limit
        if hasattr(self.retriever, 'search_kwargs'):
            self.retriever.search_kwargs['k'] = limit
        
        # Retrieve documents
        docs = self.retriever.invoke(query)
        
        # Filter by resource type if specified
        if resource_type:
            docs = [
                doc for doc in docs
                if doc.metadata.get('resource_type') == resource_type
            ]
        
        # Format results
        results = []
        for doc in docs:
            result = {
                'content': doc.page_content,
                'score': doc.metadata.get('score', 0.0),
            }
            
            if include_metadata:
                # Basic metadata
                result['metadata'] = {
                    'source_url': doc.metadata.get('source_url'),
                    'resource_type': doc.metadata.get('resource_type'),
                    'tags': doc.metadata.get('tags', []),
                    'priority': doc.metadata.get('priority'),
                }
                
                # Add citation information if available
                if (doc.metadata.get('citation_enabled') and 
                    hasattr(self.config, 'citation') and 
                    self.config.citation and 
                    self.config.citation.enabled):
                    from .citations import CitationFormatter
                    from .citations.models import SourceReference
                    
                    # Extract citation reference
                    citation_ref_data = doc.metadata.get('citation_reference')
                    if citation_ref_data:
                        try:
                            source_ref = SourceReference(**citation_ref_data)
                            
                            # Format citation using appropriate strategy
                            formatter = CitationFormatter()
                            strategy_name = doc.metadata.get('citation_strategy', 'default_citation')
                            
                            formatted_citation = formatter.format_citation(
                                source_ref=source_ref,
                                verbosity=self.config.citation.verbosity,
                                format_style=self.config.citation.format,
                                strategy_name=strategy_name
                            )
                            
                            # Add citation information to result
                            result['citation'] = {
                                'formatted': formatted_citation,
                                'source_reference': citation_ref_data,
                                'verbosity': doc.metadata.get('citation_verbosity'),
                                'strategy': strategy_name
                            }
                            
                            # Add citation link if available
                            strategy_registry = getattr(self, '_citation_strategy_registry', None)
                            if not strategy_registry:
                                from .citations.strategies import CitationStrategyRegistry
                                self._citation_strategy_registry = CitationStrategyRegistry()
                                strategy_registry = self._citation_strategy_registry
                            
                            strategy = strategy_registry.get_strategy(strategy_name)
                            citation_link = strategy.generate_citation_link(source_ref)
                            if citation_link:
                                result['citation']['link'] = citation_link
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to format citation: {e}")
                            # Add basic citation fallback
                            result['citation'] = {
                                'formatted': f"[{doc.metadata.get('source_url', 'Unknown source')}]",
                                'error': str(e)
                            }
            
            results.append(result)
        
        self.logger.debug(f"Found {len(results)} results")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        if not self.vectorstore:
            return {
                'total_documents': 0,
                'vectorstore_type': self.config.vectorstore.type,
                'embedding_model': self.config.embedding.model_name,
                'status': 'empty'
            }
        
        # Try to get document count
        try:
            if hasattr(self.vectorstore, '_collection'):
                # Chroma
                total_docs = self.vectorstore._collection.count()
            elif hasattr(self.vectorstore, 'index'):
                # FAISS
                total_docs = self.vectorstore.index.ntotal
            else:
                total_docs = "unknown"
        except:
            total_docs = "unknown"
        
        return {
            'total_documents': total_docs,
            'vectorstore_type': self.config.vectorstore.type,
            'embedding_model': self.config.embedding.model_name,
            'retriever_type': self.config.retriever.type,
            'status': 'active'
        }
    
    def clear_knowledge_base(self) -> None:
        """Clear the entire knowledge base."""
        self.logger.warning("Clearing knowledge base")
        
        if self.config.vectorstore.persist_directory:
            import shutil
            if self.config.vectorstore.persist_directory.exists():
                shutil.rmtree(self.config.vectorstore.persist_directory)
                self.logger.info("Deleted persistent storage")
        
        self.vectorstore = None
        self.retriever = None
        
        self.logger.success("Knowledge base cleared")


def create_resource_from_url(
    url: str,
    resource_type: ResourceType,
    **kwargs
) -> ResourceConfig:
    """Create a ResourceConfig from URL with smart defaults."""
    
    # Get default loader config based on URL
    loader_config = get_default_loader_config(url)
    
    # Override with any provided loader params
    if 'loader_params' in kwargs:
        loader_config.params.update(kwargs['loader_params'])
    
    # Get default splitter config based on loader type
    splitter_config = get_default_splitter_config(loader_config.type)
    
    # Override with any provided splitter params
    if 'splitter_params' in kwargs:
        splitter_config.params.update(kwargs['splitter_params'])
    
    # Override chunk settings if provided
    if 'chunk_size' in kwargs:
        splitter_config.chunk_size = kwargs['chunk_size']
    if 'chunk_overlap' in kwargs:
        splitter_config.chunk_overlap = kwargs['chunk_overlap']
    
    return ResourceConfig(
        url=url,
        resource_type=resource_type,
        loader=loader_config,
        splitter=splitter_config,
        priority=kwargs.get('priority', 1),
        tags=kwargs.get('tags', []),
        metadata=kwargs.get('metadata', {})
    ) 
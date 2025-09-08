"""
RAG Pipeline processor.

Main orchestrator for the RAG pipeline that coordinates document loading,
processing, embedding, and storage following LangChain patterns.
"""

from typing import Dict, Any, List, Optional, Tuple
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
    RAGConfig, ResourceConfig, ResourceType,
    get_default_loader_config, get_default_splitter_config
)
from .loaders import DocumentLoaderFactory
from .splitters import TextSplitterFactory
from .embeddings import EmbeddingsFactory
from .vectorstore import VectorStoreFactory
from .retrievers import RetrieverFactory
from ...utils.logging import get_logger


class ValidationError(Exception):
    """Raised when resource validation fails."""
    pass


class ProcessingError(Exception):
    """Raised when document processing fails."""
    pass


class ConfigurationError(Exception):
    """Raised when RAG configuration is invalid or missing required settings."""
    pass


class RAGProcessor:
    """Main RAG pipeline processor."""
    
    def __init__(self, config: RAGConfig):
        """Initialize the RAG processor with configuration."""
        self.config = config
        self.logger = get_logger("paas_ai.rag.pipeline")
        
        # Initialize components with proper error handling
        try:
            self.embeddings = EmbeddingsFactory.create_embeddings(config.embedding)
        except Exception as e:
            self._handle_initialization_error(e, config)
            
        self.vectorstore = None
        self.retriever = None
        
        # Load existing vectorstore if available
        self._load_existing_vectorstore()
    
    def _handle_initialization_error(self, error: Exception, config: RAGConfig):
        """Handle initialization errors with helpful messages."""
        error_str = str(error).lower()
        
        # OpenAI API key missing
        if "api_key" in error_str and ("openai" in error_str or "client option" in error_str):
            if config.embedding.type == "openai":
                raise ConfigurationError(
                    f"OpenAI API key is required for '{config.embedding.type}' embeddings.\n"
                    f"Solutions:\n"
                    f"  1. Set environment variable: export OPENAI_API_KEY='your-key-here'\n"
                    f"  2. Use local config instead: --config-profile local\n"
                    f"  3. Switch to a different embedding type in your config"
                ) from error
        
        # Azure OpenAI configuration missing
        elif "azure" in error_str:
            raise ConfigurationError(
                f"Azure OpenAI configuration is incomplete.\n"
                f"Required: azure_endpoint, api_key, and api_version\n"
                f"Check your configuration or use a different profile"
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
    
    def process_resource(self, resource: ResourceConfig) -> List[Document]:
        """Process a single resource and return documents."""
        self.logger.info(f"Processing resource: {resource.url}")
        
        try:
            # Validate resource
            self.validate_resource(resource)
            
            # Create loader
            loader = DocumentLoaderFactory.create_loader(resource.loader, resource.url)
            
            # Load documents
            self.logger.debug("Loading documents...")
            documents = loader.load()
            
            if not documents:
                self.logger.warning(f"No documents loaded from {resource.url}")
                return []
            
            self.logger.debug(f"Loaded {len(documents)} documents")
            
            # Create splitter
            splitter = TextSplitterFactory.create_splitter(resource.splitter)
            
            # Split documents
            self.logger.debug("Splitting documents...")
            # Handle splitters that only have split_text method
            if hasattr(splitter, 'split_documents'):
                split_docs = splitter.split_documents(documents)
            else:
                # For splitters like MarkdownHeaderTextSplitter that only have split_text
                # and return Document objects directly
                split_docs = []
                for doc in documents:
                    split_results = splitter.split_text(doc.page_content)
                    # Check if split_text returns Document objects or strings
                    if split_results and hasattr(split_results[0], 'page_content'):
                        # Returns Document objects
                        for split_doc in split_results:
                            # Merge original metadata with split metadata
                            merged_metadata = doc.metadata.copy()
                            merged_metadata.update(split_doc.metadata)
                            split_doc.metadata = merged_metadata
                            split_docs.append(split_doc)
                    else:
                        # Returns strings
                        for text in split_results:
                            split_doc = Document(
                                page_content=text,
                                metadata=doc.metadata.copy()
                            )
                            split_docs.append(split_doc)
            
            self.logger.debug(f"Split into {len(split_docs)} chunks")
            
            # Add metadata
            for doc in split_docs:
                doc.metadata.update({
                    'source_url': resource.url,
                    'resource_type': resource.resource_type,
                    'priority': resource.priority,
                    'tags': resource.tags,
                    'processed_at': time.time(),
                    **resource.metadata
                })
            
            return split_docs
            
        except Exception as e:
            if self.config.skip_invalid_docs:
                self.logger.error(f"Failed to process resource {resource.url}: {e}")
                return []
            else:
                raise ProcessingError(f"Failed to process resource {resource.url}: {e}")
    
    def add_resources(self, resources: List[ResourceConfig]) -> Dict[str, Any]:
        """Add multiple resources to the knowledge base."""
        self.logger.info(f"Adding {len(resources)} resources to knowledge base")
        
        results = {
            'total_resources': len(resources),
            'successful': 0,
            'failed': 0,
            'total_documents': 0,
            'errors': []
        }
        
        all_documents = []
        
        for i, resource in enumerate(resources, 1):
            self.logger.progress(f"Processing resource {i}/{len(resources)}: {resource.url}")
            
            try:
                documents = self.process_resource(resource)
                if documents:
                    all_documents.extend(documents)
                    results['total_documents'] += len(documents)
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"No documents from {resource.url}")
                    
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"{resource.url}: {str(e)}")
                self.logger.error(f"Failed to process {resource.url}: {e}")
        
        # Add documents to vectorstore
        if all_documents:
            self.logger.info(f"Adding {len(all_documents)} documents to vectorstore")
            self._add_documents_to_vectorstore(all_documents)
        
        self.logger.success(
            f"Completed processing. "
            f"Successful: {results['successful']}, "
            f"Failed: {results['failed']}, "
            f"Total documents: {results['total_documents']}"
        )
        
        return results
    
    def _add_documents_to_vectorstore(self, documents: List[Document]) -> None:
        """Add documents to the vectorstore."""
        # Filter complex metadata for compatibility with vectorstores like Chroma
        from langchain_community.vectorstores.utils import filter_complex_metadata
        filtered_documents = filter_complex_metadata(documents)
        
        if not self.vectorstore:
            # Create new vectorstore
            self.vectorstore = VectorStoreFactory.create_vectorstore(
                self.config.vectorstore,
                self.embeddings,
                filtered_documents
            )
            self.logger.info("Created new vectorstore")
        else:
            # Add to existing vectorstore
            self.vectorstore.add_documents(filtered_documents)
            self.logger.info(f"Added {len(filtered_documents)} documents to existing vectorstore")
        
        # Create/update retriever
        self.retriever = RetrieverFactory.create_retriever(
            self.config.retriever,
            self.vectorstore
        )
    
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
                result['metadata'] = {
                    'source_url': doc.metadata.get('source_url'),
                    'resource_type': doc.metadata.get('resource_type'),
                    'tags': doc.metadata.get('tags', []),
                    'priority': doc.metadata.get('priority'),
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
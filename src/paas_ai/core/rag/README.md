# RAG (Retrieval-Augmented Generation) System

A comprehensive RAG implementation following LangChain patterns and best practices.

## Overview

This RAG system provides a complete pipeline for:
- Loading documents from various sources (web, files, APIs)
- Intelligent text splitting based on content type
- Flexible embedding generation
- Multiple vector store backends
- Advanced retrieval strategies including ensemble methods

## Key Features

- **Configurable Components**: Factory pattern for all components
- **Smart Defaults**: Automatic loader/splitter selection based on URL/content type
- **Multiple Backends**: Support for Chroma, FAISS, Pinecone vector stores
- **Ensemble Retrieval**: Combine multiple retrieval strategies
- **Input Validation**: URL validation and error handling
- **Batch Processing**: Efficient processing of multiple resources
- **Persistent Storage**: Save and load vector stores

## Quick Start

### Basic Usage

```python
from paas_ai.core.rag import RAGProcessor, create_resource_from_url
from paas_ai.core.rag.config import DEFAULT_CONFIGS, ResourceType

# Initialize processor with default config
processor = RAGProcessor(DEFAULT_CONFIGS['default'])

# Add a single resource
resource = create_resource_from_url(
    "https://kubernetes.io/docs/concepts/",
    ResourceType.DSL,
    tags=["kubernetes", "official"],
    priority=8
)

# Process the resource
results = processor.add_resources([resource])

# Search the knowledge base
search_results = processor.search("kubernetes deployment best practices")
```

### CLI Usage

```bash
# Add a single resource
paas-ai rag resources add --url "https://kubernetes.io/docs" --type dsl

# Add resources from CSV file
paas-ai rag resources add --csv-file resources.csv

# Search the knowledge base
paas-ai rag search "microservice patterns"

# Check system status
paas-ai rag status --detailed
```

## Configuration

### Profiles

Three built-in configuration profiles are available:

- **default**: OpenAI embeddings + Chroma vector store
- **local**: SentenceTransformers + FAISS (no API keys needed)
- **production**: OpenAI large embeddings + Pinecone + ensemble retrieval

### Custom Configuration

```python
from paas_ai.core.rag.config import RAGConfig, EmbeddingConfig, VectorStoreConfig

config = RAGConfig(
    embedding=EmbeddingConfig(
        type="openai",
        model_name="text-embedding-3-small"
    ),
    vectorstore=VectorStoreConfig(
        type="chroma",
        persist_directory=Path("./my_rag_data"),
        collection_name="my_knowledge_base"
    ),
    batch_size=64,
    validate_urls=True
)
```

## Supported Loaders

- **Web**: Any HTTP/HTTPS URL
- **PDF**: Local and remote PDF files
- **Markdown**: .md and .markdown files
- **HTML**: Local and remote HTML files
- **JSON**: Structured JSON documents
- **CSV**: Comma-separated value files
- **Directory**: Local directories with multiple files
- **GitHub**: GitHub repositories and issues
- **Confluence**: Confluence pages (requires credentials)
- **Notion**: Notion pages (requires credentials)

## Supported Text Splitters

- **Character**: Simple character-based splitting
- **Recursive Character**: Smart recursive splitting (recommended)
- **Markdown**: Markdown-aware splitting with header preservation
- **HTML**: HTML-aware splitting with tag preservation
- **JSON**: JSON structure-aware splitting
- **Code**: Programming language-aware splitting
- **Token**: Token-based splitting for precise control

## Supported Embeddings

- **OpenAI**: `text-embedding-3-small`, `text-embedding-3-large`
- **Azure OpenAI**: Azure-hosted OpenAI models
- **HuggingFace**: Any HuggingFace embedding model
- **Sentence Transformers**: Local sentence transformer models
- **Cohere**: Cohere embedding models

## Supported Vector Stores

- **Chroma**: Local persistent vector store
- **FAISS**: High-performance similarity search
- **Pinecone**: Cloud vector database
- **Qdrant**: Vector similarity search engine
- **PGVector**: PostgreSQL extension for vectors

## Supported Retrievers

- **Similarity**: Standard cosine similarity search
- **MMR**: Maximum marginal relevance for diversity
- **Score Threshold**: Filter by similarity score
- **Ensemble**: Combine multiple retrieval strategies
- **Multi-Query**: Generate multiple queries for comprehensive search
- **Parent Document**: Retrieve full documents from smaller chunks

## CSV Format for Bulk Resource Addition

```csv
url,type,priority,tags,chunk_size,chunk_overlap
https://kubernetes.io/docs/,dsl,8,"kubernetes,official",1500,300
https://docs.docker.com/,contextual,6,"docker,containers",1200,200
./docs/security.md,guidelines,9,"security,internal",1000,200
```

Required columns:
- `url`: Resource URL or file path
- `type`: Resource type (dsl, contextual, guidelines, domain_rules)

Optional columns:
- `priority`: Priority 1-10 (default: 1)
- `tags`: Comma-separated tags
- `chunk_size`: Custom chunk size
- `chunk_overlap`: Custom chunk overlap

## Error Handling

The system includes comprehensive error handling:

- **Validation Errors**: Invalid URLs or configurations
- **Processing Errors**: Failed document loading or splitting
- **Network Errors**: Connection timeouts or HTTP errors
- **Configuration Errors**: Invalid component configurations

Configure error handling behavior:

```python
config = RAGConfig(
    validate_urls=True,       # Validate URLs before processing
    skip_invalid_docs=True,   # Continue processing if some docs fail
    max_retries=3,           # Retry failed operations
    timeout=300              # Timeout for operations
)
```

## Performance Optimization

### Batch Processing

Process multiple resources efficiently:

```python
# Process in batches
config.batch_size = 64  # Larger batches for better throughput

# Parallel processing will be added in future versions
```

### Memory Management

For large document sets:

- Use smaller chunk sizes to reduce memory usage
- Consider FAISS for very large vector stores
- Use persistent storage to avoid reprocessing

### Search Optimization

Optimize search performance:

- Use appropriate `k` values for retrieval
- Consider ensemble retrieval for better quality
- Use score thresholds to filter irrelevant results

## Integration Examples

### With LangChain Applications

```python
# Use as a retriever in LangChain chains
retriever = processor.retriever

# Integration with QA chains
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=retriever
)

result = qa_chain.invoke({"query": "How to deploy microservices?"})
```

### Custom Applications

```python
# Custom search with post-processing
results = processor.search("kubernetes security", limit=10)
filtered_results = [r for r in results if r['score'] > 0.8]

# Add custom metadata
for result in filtered_results:
    result['custom_field'] = process_content(result['content'])
```

## Development and Testing

### Running Tests

```bash
# Run unit tests
pytest tests/unit/test_core/test_rag/

# Run integration tests
pytest tests/integration/test_rag/

# Run all RAG tests
pytest -k "rag"
```

### Local Development

For development without API keys:

```python
# Use local configuration
processor = RAGProcessor(DEFAULT_CONFIGS['local'])
```

This uses SentenceTransformers for embeddings and FAISS for vector storage, requiring no external API keys.

## Troubleshooting

### Common Issues

1. **Empty Knowledge Base**: Add resources using the CLI or API
2. **No Search Results**: Check if documents were processed successfully
3. **Memory Issues**: Reduce batch size or use smaller chunk sizes
4. **API Errors**: Verify API keys and rate limits

### Debug Mode

Enable verbose logging:

```python
config.verbose = True
config.log_level = "DEBUG"
```

### Status Checking

```bash
# Check system status
paas-ai rag status --detailed

# View configuration
paas-ai rag status --format json
```

## Future Enhancements

- Async processing for better performance
- Advanced preprocessing pipelines
- Custom retrieval strategies
- Multi-modal document support
- Real-time document updates
- Advanced analytics and monitoring 
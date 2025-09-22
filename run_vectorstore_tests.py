#!/usr/bin/env python3
"""
Test runner for vectorstore tests with comprehensive mocking of missing dependencies.
"""

import sys
import os
import unittest.mock as mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock all the problematic dependencies
def mock_dependencies():
    """Mock all problematic dependencies."""
    
    # Mock crawl4ai
    crawl4ai_mock = mock.MagicMock()
    crawl4ai_mock.AsyncWebCrawler = mock.MagicMock()
    crawl4ai_mock.BrowserConfig = mock.MagicMock()
    crawl4ai_mock.CrawlerRunConfig = mock.MagicMock()
    crawl4ai_mock.CacheMode = mock.MagicMock()
    
    models_mock = mock.MagicMock()
    models_mock.CrawlResult = mock.MagicMock()
    crawl4ai_mock.models = models_mock
    
    sys.modules['crawl4ai'] = crawl4ai_mock
    sys.modules['crawl4ai.models'] = models_mock
    
    # Mock langchain_huggingface
    langchain_huggingface_mock = mock.MagicMock()
    langchain_huggingface_mock.HuggingFaceEmbeddings = mock.MagicMock()
    sys.modules['langchain_huggingface'] = langchain_huggingface_mock
    
    # Mock other langchain modules
    sys.modules['langchain_chroma'] = mock.MagicMock()
    sys.modules['langchain_community'] = mock.MagicMock()
    sys.modules['langchain_community.vectorstores'] = mock.MagicMock()
    sys.modules['langchain_pinecone'] = mock.MagicMock()
    
    # Mock other dependencies
    sys.modules['faiss'] = mock.MagicMock()
    sys.modules['numpy'] = mock.MagicMock()
    sys.modules['numpy'] = mock.MagicMock()
    
    # Mock other langchain modules
    sys.modules['langchain'] = mock.MagicMock()
    sys.modules['langchain.embeddings'] = mock.MagicMock()
    sys.modules['langchain.vectorstores'] = mock.MagicMock()
    sys.modules['langchain.text_splitter'] = mock.MagicMock()
    sys.modules['langchain.document_loaders'] = mock.MagicMock()
    
    # Mock other dependencies that might be imported
    sys.modules['requests'] = mock.MagicMock()
    sys.modules['urllib'] = mock.MagicMock()
    sys.modules['urllib.parse'] = mock.MagicMock()
    sys.modules['urllib.request'] = mock.MagicMock()
    sys.modules['urllib.error'] = mock.MagicMock()
    sys.modules['json'] = mock.MagicMock()
    sys.modules['yaml'] = mock.MagicMock()
    sys.modules['pathlib'] = mock.MagicMock()
    sys.modules['os'] = mock.MagicMock()
    sys.modules['tempfile'] = mock.MagicMock()
    sys.modules['shutil'] = mock.MagicMock()
    sys.modules['typing'] = mock.MagicMock()
    sys.modules['abc'] = mock.MagicMock()
    sys.modules['enum'] = mock.MagicMock()
    sys.modules['dataclasses'] = mock.MagicMock()
    sys.modules['functools'] = mock.MagicMock()
    sys.modules['itertools'] = mock.MagicMock()
    sys.modules['collections'] = mock.MagicMock()
    sys.modules['copy'] = mock.MagicMock()
    sys.modules['warnings'] = mock.MagicMock()
    sys.modules['logging'] = mock.MagicMock()
    sys.modules['inspect'] = mock.MagicMock()
    sys.modules['importlib'] = mock.MagicMock()
    sys.modules['importlib.util'] = mock.MagicMock()
    sys.modules['importlib.metadata'] = mock.MagicMock()
    sys.modules['pkg_resources'] = mock.MagicMock()
    sys.modules['setuptools'] = mock.MagicMock()
    sys.modules['distutils'] = mock.MagicMock()
    sys.modules['distutils.util'] = mock.MagicMock()
    sys.modules['distutils.version'] = mock.MagicMock()
    sys.modules['distutils.errors'] = mock.MagicMock()
    sys.modules['distutils.spawn'] = mock.MagicMock()
    sys.modules['distutils.file_util'] = mock.MagicMock()
    sys.modules['distutils.dir_util'] = mock.MagicMock()
    sys.modules['distutils.archive_util'] = mock.MagicMock()
    sys.modules['distutils.dep_util'] = mock.MagicMock()
    sys.modules['distutils.util'] = mock.MagicMock()
    sys.modules['distutils.version'] = mock.MagicMock()
    sys.modules['distutils.errors'] = mock.MagicMock()
    sys.modules['distutils.spawn'] = mock.MagicMock()
    sys.modules['distutils.file_util'] = mock.MagicMock()
    sys.modules['distutils.dir_util'] = mock.MagicMock()
    sys.modules['distutils.archive_util'] = mock.MagicMock()
    sys.modules['distutils.dep_util'] = mock.MagicMock()
    sys.modules['distutils.util'] = mock.MagicMock()
    sys.modules['distutils.version'] = mock.MagicMock()
    sys.modules['distutils.errors'] = mock.MagicMock()
    sys.modules['distutils.spawn'] = mock.MagicMock()
    sys.modules['distutils.file_util'] = mock.MagicMock()
    sys.modules['distutils.dir_util'] = mock.MagicMock()
    sys.modules['distutils.archive_util'] = mock.MagicMock()
    sys.modules['distutils.dep_util'] = mock.MagicMock()
    
    # Mock langchain_core
    langchain_core_mock = mock.MagicMock()
    langchain_core_mock.documents = mock.MagicMock()
    langchain_core_mock.documents.Document = mock.MagicMock()
    langchain_core_mock.embeddings = mock.MagicMock()
    langchain_core_mock.embeddings.Embeddings = mock.MagicMock()
    langchain_core_mock.vectorstores = mock.MagicMock()
    langchain_core_mock.vectorstores.VectorStore = mock.MagicMock()
    langchain_core_mock.retrievers = mock.MagicMock()
    langchain_core_mock.retrievers.BaseRetriever = mock.MagicMock()
    sys.modules['langchain_core'] = langchain_core_mock
    sys.modules['langchain_core.documents'] = langchain_core_mock.documents
    sys.modules['langchain_core.embeddings'] = langchain_core_mock.embeddings
    sys.modules['langchain_core.vectorstores'] = langchain_core_mock.vectorstores
    sys.modules['langchain_core.retrievers'] = langchain_core_mock.retrievers

if __name__ == "__main__":
    # Mock dependencies first
    mock_dependencies()
    
    # Now run the tests
    import pytest
    
    # Run the vectorstore tests
    test_dir = os.path.join(os.path.dirname(__file__), 'tests', 'unit', 'test_core', 'test_rag', 'test_vectorstore')
    exit_code = pytest.main([test_dir, '-v'])
    sys.exit(exit_code)

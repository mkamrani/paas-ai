"""
Unit tests for Crawl4AI web loader strategy and implementation.

Tests all components of the Crawl4AI web loader system including:
- Crawl4AIWebLoaderStrategy class
- Crawl4AIWebLoader implementation
- Configuration validation
- URL handling and validation
- Error handling and edge cases
- Async functionality
- CSV file processing
"""

import asyncio
import csv
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from contextlib import contextmanager

from src.paas_ai.core.rag.loaders.crawl4ai_web import Crawl4AIWebLoaderStrategy
from src.paas_ai.core.rag.loaders.crawl4ai_web_loader import Crawl4AIWebLoader
from src.paas_ai.core.rag.config import LoaderConfig, LoaderType


@contextmanager
def temp_csv_file(content: str):
    """Context manager for creating temporary CSV files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        f.flush()
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


class TestCrawl4AIWebLoaderStrategy:
    """Test the Crawl4AIWebLoaderStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = Crawl4AIWebLoaderStrategy()
        assert strategy is not None
    
    def test_create_loader_default_config(self):
        """Test creating loader with default configuration."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = "https://example.com"
        
        with patch('src.paas_ai.core.rag.loaders.crawl4ai_web.Crawl4AIWebLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            
            result = strategy.create_loader(config, url)
            
            mock_loader_class.assert_called_once_with(
                web_paths=[url],
                headless=True,
                wait_time=3.0,
                timeout=30000,
                enable_stealth=False
            )
            assert result == mock_loader
    
    def test_create_loader_custom_config(self):
        """Test creating loader with custom configuration."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(
            type=LoaderType.CRAWL4AI_WEB,
            params={
                'headless': False,
                'wait_time': 5.0,
                'timeout': 60000,
                'enable_stealth': True
            }
        )
        url = "https://example.com"
        
        with patch('src.paas_ai.core.rag.loaders.crawl4ai_web.Crawl4AIWebLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            
            result = strategy.create_loader(config, url)
            
            mock_loader_class.assert_called_once_with(
                web_paths=[url],
                headless=False,
                wait_time=5.0,
                timeout=60000,
                enable_stealth=True
            )
            assert result == mock_loader
    
    def test_create_loader_partial_config(self):
        """Test creating loader with partial configuration."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(
            type=LoaderType.CRAWL4AI_WEB,
            params={'headless': False}
        )
        url = "https://example.com"
        
        with patch('src.paas_ai.core.rag.loaders.crawl4ai_web.Crawl4AIWebLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            
            result = strategy.create_loader(config, url)
            
            mock_loader_class.assert_called_once_with(
                web_paths=[url],
                headless=False,
                wait_time=3.0,  # Default value
                timeout=30000,  # Default value
                enable_stealth=False  # Default value
            )
            assert result == mock_loader
    
    def test_validate_config_valid_url(self):
        """Test configuration validation with valid URL."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = "https://example.com"
        
        # Should not raise any exception
        strategy.validate_config(config, url)
    
    def test_validate_config_http_url(self):
        """Test configuration validation with HTTP URL."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = "http://example.com"
        
        # Should not raise any exception
        strategy.validate_config(config, url)
    
    def test_validate_config_csv_url(self):
        """Test configuration validation with CSV URL."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = "urls.csv"
        
        # Should not raise any exception
        strategy.validate_config(config, url)
    
    def test_validate_config_empty_url(self):
        """Test configuration validation with empty URL."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = ""
        
        with pytest.raises(ValueError, match="URL is required for Crawl4AI web loader"):
            strategy.validate_config(config, url)
    
    def test_validate_config_none_url(self):
        """Test configuration validation with None URL."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = None
        
        with pytest.raises(ValueError, match="URL is required for Crawl4AI web loader"):
            strategy.validate_config(config, url)
    
    def test_validate_config_invalid_url(self):
        """Test configuration validation with invalid URL."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = "not-a-url"
        
        with pytest.raises(ValueError, match="Invalid URL format for Crawl4AI web loader"):
            strategy.validate_config(config, url)
    
    def test_validate_config_invalid_timeout(self):
        """Test configuration validation with invalid timeout."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(
            type=LoaderType.CRAWL4AI_WEB,
            params={'timeout': -1}
        )
        url = "https://example.com"
        
        with pytest.raises(ValueError, match="Invalid timeout value"):
            strategy.validate_config(config, url)
    
    def test_validate_config_invalid_timeout_type(self):
        """Test configuration validation with invalid timeout type."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(
            type=LoaderType.CRAWL4AI_WEB,
            params={'timeout': "invalid"}
        )
        url = "https://example.com"
        
        with pytest.raises(ValueError, match="Invalid timeout value"):
            strategy.validate_config(config, url)
    
    def test_validate_config_invalid_wait_time(self):
        """Test configuration validation with invalid wait_time."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(
            type=LoaderType.CRAWL4AI_WEB,
            params={'wait_time': -1}
        )
        url = "https://example.com"
        
        with pytest.raises(ValueError, match="Invalid wait_time value"):
            strategy.validate_config(config, url)
    
    def test_validate_config_invalid_wait_time_type(self):
        """Test configuration validation with invalid wait_time type."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(
            type=LoaderType.CRAWL4AI_WEB,
            params={'wait_time': "invalid"}
        )
        url = "https://example.com"
        
        with pytest.raises(ValueError, match="Invalid wait_time value"):
            strategy.validate_config(config, url)
    
    def test_validate_config_valid_timeout(self):
        """Test configuration validation with valid timeout."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(
            type=LoaderType.CRAWL4AI_WEB,
            params={'timeout': 45000}
        )
        url = "https://example.com"
        
        # Should not raise any exception
        strategy.validate_config(config, url)
    
    def test_validate_config_valid_wait_time(self):
        """Test configuration validation with valid wait_time."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(
            type=LoaderType.CRAWL4AI_WEB,
            params={'wait_time': 2.5}
        )
        url = "https://example.com"
        
        # Should not raise any exception
        strategy.validate_config(config, url)


class TestCrawl4AIWebLoader:
    """Test the Crawl4AIWebLoader class."""
    
    def test_init_defaults(self):
        """Test loader initialization with default values."""
        loader = Crawl4AIWebLoader()
        
        assert loader.web_paths == []
        assert loader.headless is True
        assert loader.wait_time == 3.0
        assert loader.timeout == 30000
        assert loader.enable_stealth is True
    
    def test_init_with_web_paths(self):
        """Test loader initialization with web paths."""
        web_paths = ["https://example.com", "https://test.com"]
        loader = Crawl4AIWebLoader(web_paths=web_paths)
        
        assert loader.web_paths == web_paths
        assert loader.headless is True
        assert loader.wait_time == 3.0
        assert loader.timeout == 30000
        assert loader.enable_stealth is True
    
    def test_init_with_web_path(self):
        """Test loader initialization with single web path."""
        web_path = "https://example.com"
        loader = Crawl4AIWebLoader(web_path=web_path)
        
        assert loader.web_paths == [web_path]
        assert loader.headless is True
        assert loader.wait_time == 3.0
        assert loader.timeout == 30000
        assert loader.enable_stealth is True
    
    def test_init_custom_params(self):
        """Test loader initialization with custom parameters."""
        web_paths = ["https://example.com"]
        loader = Crawl4AIWebLoader(
            web_paths=web_paths,
            headless=False,
            wait_time=5.0,
            timeout=60000,
            enable_stealth=False
        )
        
        assert loader.web_paths == web_paths
        assert loader.headless is False
        assert loader.wait_time == 5.0
        assert loader.timeout == 60000
        assert loader.enable_stealth is False
    
    def test_browser_config_creation(self):
        """Test browser configuration creation."""
        loader = Crawl4AIWebLoader(
            headless=False,
            enable_stealth=True
        )
        
        assert loader.browser_config.headless is False
        assert loader.browser_config.enable_stealth is True
        assert loader.browser_config.viewport_width == 1920
        assert loader.browser_config.viewport_height == 1080
        assert loader.browser_config.verbose is True
    
    def test_crawler_config_creation(self):
        """Test crawler configuration creation."""
        loader = Crawl4AIWebLoader(
            wait_time=5.0,
            timeout=60000
        )
        
        assert loader.crawler_config.page_timeout == 60000
        assert loader.crawler_config.delay_before_return_html == 5.0
        assert loader.crawler_config.wait_for == "css:body"
        assert loader.crawler_config.remove_overlay_elements is True
        assert loader.crawler_config.verbose is True
    
    def test_extract_title_from_markdown_with_title(self):
        """Test title extraction from markdown with title."""
        loader = Crawl4AIWebLoader()
        content = "# Test Title\n\nSome content here."
        
        title = loader._extract_title_from_markdown(content)
        assert title == "Test Title"
    
    def test_extract_title_from_markdown_without_title(self):
        """Test title extraction from markdown without title."""
        loader = Crawl4AIWebLoader()
        content = "Some content without title."
        
        title = loader._extract_title_from_markdown(content)
        assert title == "Unknown Title"
    
    def test_extract_title_from_markdown_multiple_headers(self):
        """Test title extraction from markdown with multiple headers."""
        loader = Crawl4AIWebLoader()
        content = "## Second Header\n\n# First Header\n\nSome content."
        
        title = loader._extract_title_from_markdown(content)
        assert title == "First Header"
    
    def test_extract_title_from_markdown_empty_content(self):
        """Test title extraction from empty markdown content."""
        loader = Crawl4AIWebLoader()
        content = ""
        
        title = loader._extract_title_from_markdown(content)
        assert title == "Unknown Title"
    
    def test_parse_csv_urls_with_headers(self):
        """Test parsing CSV URLs with headers."""
        csv_content = "url,title,description\nhttps://example.com,Example,Test site\nhttps://test.com,Test,Another site"
        
        with temp_csv_file(csv_content) as csv_path:
            loader = Crawl4AIWebLoader()
            urls = loader._parse_csv_urls(str(csv_path))
            
            assert urls == ["https://example.com", "https://test.com"]
    
    def test_parse_csv_urls_without_headers(self):
        """Test parsing CSV URLs without headers."""
        csv_content = "https://example.com\nhttps://test.com\nhttps://another.com"
        
        with temp_csv_file(csv_content) as csv_path:
            loader = Crawl4AIWebLoader()
            urls = loader._parse_csv_urls(str(csv_path))
            
            assert urls == ["https://example.com", "https://test.com", "https://another.com"]
    
    def test_parse_csv_urls_different_url_column(self):
        """Test parsing CSV URLs with different URL column name."""
        csv_content = "link,title,description\nhttps://example.com,Example,Test site\nhttps://test.com,Test,Another site"
        
        with temp_csv_file(csv_content) as csv_path:
            loader = Crawl4AIWebLoader()
            urls = loader._parse_csv_urls(str(csv_path))
            
            assert urls == ["https://example.com", "https://test.com"]
    
    def test_parse_csv_urls_invalid_urls_filtered(self):
        """Test that invalid URLs are filtered out."""
        csv_content = "url,title\nhttps://example.com,Valid\nnot-a-url,Invalid\nhttps://test.com,Valid"
        
        with temp_csv_file(csv_content) as csv_path:
            loader = Crawl4AIWebLoader()
            urls = loader._parse_csv_urls(str(csv_path))
            
            assert urls == ["https://example.com", "https://test.com"]
    
    def test_parse_csv_urls_no_url_column(self):
        """Test parsing CSV without URL column."""
        csv_content = "title,description\nExample,Test site\nTest,Another site"
        
        with temp_csv_file(csv_content) as csv_path:
            loader = Crawl4AIWebLoader()
            
            # The current implementation doesn't raise an error, it just returns empty list
            # This is the actual behavior, so we test for that
            urls = loader._parse_csv_urls(str(csv_path))
            assert urls == []
    
    def test_parse_csv_urls_file_not_found(self):
        """Test parsing CSV file that doesn't exist."""
        loader = Crawl4AIWebLoader()
        
        with pytest.raises(FileNotFoundError):
            loader._parse_csv_urls("nonexistent.csv")
    
    def test_parse_csv_urls_empty_file(self):
        """Test parsing empty CSV file."""
        csv_content = ""
        
        with temp_csv_file(csv_content) as csv_path:
            loader = Crawl4AIWebLoader()
            
            # Empty files should raise an error due to CSV sniffer issues
            with pytest.raises(Exception):
                loader._parse_csv_urls(str(csv_path))
    
    @pytest.mark.asyncio
    async def test_load_single_url_success(self):
        """Test loading content from a single URL successfully."""
        loader = Crawl4AIWebLoader()
        url = "https://example.com"
        
        # Mock the CrawlResult
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "# Test Title\n\nTest content here."
        mock_result.error_message = None
        
        with patch('src.paas_ai.core.rag.loaders.crawl4ai_web_loader.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
            mock_crawler.arun.return_value = mock_result
            
            document = await loader._load_single_url(url)
            
            assert document.page_content == "# Test Title\n\nTest content here."
            assert document.metadata["source"] == url
            assert document.metadata["url"] == url
            assert document.metadata["title"] == "Test Title"
            assert document.metadata["content_length"] == len("# Test Title\n\nTest content here.")
            assert document.metadata["success"] is True
            assert document.metadata["loader_type"] == "crawl4ai_web"
    
    @pytest.mark.asyncio
    async def test_load_single_url_failure(self):
        """Test loading content from a single URL that fails."""
        loader = Crawl4AIWebLoader()
        url = "https://example.com"
        
        # Mock the CrawlResult
        mock_result = Mock()
        mock_result.success = False
        mock_result.markdown = None
        mock_result.error_message = "Connection timeout"
        
        with patch('src.paas_ai.core.rag.loaders.crawl4ai_web_loader.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
            mock_crawler.arun.return_value = mock_result
            
            document = await loader._load_single_url(url)
            
            assert "Error loading" in document.page_content
            assert document.metadata["source"] == url
            assert document.metadata["url"] == url
            assert document.metadata["title"] == f"Error: {url}"
            assert document.metadata["content_length"] == 0
            assert document.metadata["success"] is False
            assert document.metadata["error"] == "Failed to crawl https://example.com: Connection timeout"
            assert document.metadata["loader_type"] == "crawl4ai_web"
    
    @pytest.mark.asyncio
    async def test_load_single_url_exception(self):
        """Test loading content from a single URL that raises an exception."""
        loader = Crawl4AIWebLoader()
        url = "https://example.com"
        
        with patch('src.paas_ai.core.rag.loaders.crawl4ai_web_loader.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
            mock_crawler.arun.side_effect = Exception("Network error")
            
            document = await loader._load_single_url(url)
            
            assert "Error loading" in document.page_content
            assert "Network error" in document.page_content
            assert document.metadata["source"] == url
            assert document.metadata["url"] == url
            assert document.metadata["title"] == f"Error: {url}"
            assert document.metadata["content_length"] == 0
            assert document.metadata["success"] is False
            assert document.metadata["error"] == "Network error"
            assert document.metadata["loader_type"] == "crawl4ai_web"
    
    @pytest.mark.asyncio
    async def test_load_single_url_empty_content(self):
        """Test loading content from a single URL with empty content."""
        loader = Crawl4AIWebLoader()
        url = "https://example.com"
        
        # Mock the CrawlResult
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = ""
        mock_result.error_message = None
        
        with patch('src.paas_ai.core.rag.loaders.crawl4ai_web_loader.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
            mock_crawler.arun.return_value = mock_result
            
            document = await loader._load_single_url(url)
            
            assert document.page_content == "No content extracted"
            assert document.metadata["source"] == url
            assert document.metadata["url"] == url
            assert document.metadata["title"] == "Unknown Title"
            assert document.metadata["content_length"] == len("No content extracted")
            assert document.metadata["success"] is True
            assert document.metadata["loader_type"] == "crawl4ai_web"
    
    @pytest.mark.asyncio
    async def test_load_batch_urls_success(self):
        """Test loading content from multiple URLs successfully."""
        loader = Crawl4AIWebLoader()
        urls = ["https://example.com", "https://test.com"]
        
        # Mock the CrawlResults
        mock_result1 = Mock()
        mock_result1.success = True
        mock_result1.markdown = "# Example\n\nContent 1"
        mock_result1.error_message = None
        mock_result1.url = "https://example.com"
        
        mock_result2 = Mock()
        mock_result2.success = True
        mock_result2.markdown = "# Test\n\nContent 2"
        mock_result2.error_message = None
        mock_result2.url = "https://test.com"
        
        with patch('src.paas_ai.core.rag.loaders.crawl4ai_web_loader.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
            mock_crawler.arun_many.return_value = [mock_result1, mock_result2]
            
            documents = await loader._load_batch_urls(urls)
            
            assert len(documents) == 2
            
            # Check first document
            assert documents[0].page_content == "# Example\n\nContent 1"
            assert documents[0].metadata["source"] == "https://example.com"
            assert documents[0].metadata["title"] == "Example"
            assert documents[0].metadata["success"] is True
            
            # Check second document
            assert documents[1].page_content == "# Test\n\nContent 2"
            assert documents[1].metadata["source"] == "https://test.com"
            assert documents[1].metadata["title"] == "Test"
            assert documents[1].metadata["success"] is True
    
    @pytest.mark.asyncio
    async def test_load_batch_urls_mixed_results(self):
        """Test loading content from multiple URLs with mixed success/failure."""
        loader = Crawl4AIWebLoader()
        urls = ["https://example.com", "https://test.com"]
        
        # Mock the CrawlResults
        mock_result1 = Mock()
        mock_result1.success = True
        mock_result1.markdown = "# Example\n\nContent 1"
        mock_result1.error_message = None
        mock_result1.url = "https://example.com"
        
        mock_result2 = Mock()
        mock_result2.success = False
        mock_result2.markdown = None
        mock_result2.error_message = "Connection failed"
        mock_result2.url = "https://test.com"
        
        with patch('src.paas_ai.core.rag.loaders.crawl4ai_web_loader.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
            mock_crawler.arun_many.return_value = [mock_result1, mock_result2]
            
            documents = await loader._load_batch_urls(urls)
            
            assert len(documents) == 2
            
            # Check successful document
            assert documents[0].page_content == "# Example\n\nContent 1"
            assert documents[0].metadata["success"] is True
            
            # Check failed document
            assert "Error loading" in documents[1].page_content
            assert documents[1].metadata["success"] is False
            assert documents[1].metadata["error"] == "Connection failed"
    
    @pytest.mark.asyncio
    async def test_load_batch_urls_exception(self):
        """Test loading content from multiple URLs when batch operation fails."""
        loader = Crawl4AIWebLoader()
        urls = ["https://example.com", "https://test.com"]
        
        with patch('src.paas_ai.core.rag.loaders.crawl4ai_web_loader.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
            mock_crawler.arun_many.side_effect = Exception("Batch error")
            
            documents = await loader._load_batch_urls(urls)
            
            assert len(documents) == 2
            
            # Both documents should have error content
            for i, url in enumerate(urls):
                assert "Batch error" in documents[i].page_content
                assert documents[i].metadata["source"] == url
                assert documents[i].metadata["success"] is False
                assert documents[i].metadata["error"] == "Batch error"
    
    @pytest.mark.asyncio
    async def test_aload_single_url(self):
        """Test async loading with single URL."""
        loader = Crawl4AIWebLoader(web_paths=["https://example.com"])
        
        with patch.object(loader, '_load_single_url') as mock_load_single:
            mock_document = Mock()
            mock_load_single.return_value = mock_document
            
            documents = await loader.aload()
            
            assert len(documents) == 1
            assert documents[0] == mock_document
            mock_load_single.assert_called_once_with("https://example.com")
    
    @pytest.mark.asyncio
    async def test_aload_csv_file(self):
        """Test async loading with CSV file."""
        csv_content = "url,title\nhttps://example.com,Example\nhttps://test.com,Test"
        
        with temp_csv_file(csv_content) as csv_path:
            loader = Crawl4AIWebLoader(web_paths=[str(csv_path)])
            
            with patch.object(loader, '_load_batch_urls') as mock_load_batch:
                mock_documents = [Mock(), Mock()]
                mock_load_batch.return_value = mock_documents
                
                documents = await loader.aload()
                
                assert len(documents) == 2
                assert documents == mock_documents
                mock_load_batch.assert_called_once_with(["https://example.com", "https://test.com"])
    
    @pytest.mark.asyncio
    async def test_aload_unsupported_format(self):
        """Test async loading with unsupported format."""
        loader = Crawl4AIWebLoader(web_paths=["unsupported-format"])
        
        documents = await loader.aload()
        
        assert len(documents) == 1
        assert "Unsupported source format" in documents[0].page_content
        assert documents[0].metadata["success"] is False
        assert documents[0].metadata["error"] == "Unsupported source format"
    
    @pytest.mark.asyncio
    async def test_aload_no_web_paths(self):
        """Test async loading with no web paths."""
        loader = Crawl4AIWebLoader(web_paths=[])
        
        documents = await loader.aload()
        
        assert documents == []
    
    def test_load_sync_with_running_loop(self):
        """Test synchronous load method when event loop is already running."""
        loader = Crawl4AIWebLoader(web_paths=["https://example.com"])
        
        with patch('asyncio.get_running_loop') as mock_get_loop:
            mock_loop = Mock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop
            
            with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
                mock_future = Mock()
                mock_future.result.return_value = [Mock()]
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
                
                documents = loader.load()
                
                assert len(documents) == 1
    
    def test_load_sync_without_running_loop(self):
        """Test synchronous load method when no event loop is running."""
        loader = Crawl4AIWebLoader(web_paths=["https://example.com"])
        
        with patch('asyncio.get_running_loop', side_effect=RuntimeError("No running loop")):
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = [Mock()]
                
                documents = loader.load()
                
                assert len(documents) == 1
                mock_run.assert_called_once()
    
    def test_load_sync_runtime_error(self):
        """Test synchronous load method with RuntimeError."""
        loader = Crawl4AIWebLoader(web_paths=["https://example.com"])
        
        with patch('asyncio.get_running_loop', side_effect=RuntimeError("Some error")):
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = [Mock()]
                
                documents = loader.load()
                
                assert len(documents) == 1
                mock_run.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_loader_strategy_with_none_config_params(self):
        """Test loader strategy with config that has None params."""
        strategy = Crawl4AIWebLoaderStrategy()
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        config.params = None
        url = "https://example.com"
        
        with patch('src.paas_ai.core.rag.loaders.crawl4ai_web.Crawl4AIWebLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            
            result = strategy.create_loader(config, url)
            
            # Should use empty dict as fallback
            mock_loader_class.assert_called_once_with(
                web_paths=[url],
                headless=True,
                wait_time=3.0,
                timeout=30000,
                enable_stealth=False
            )
    
    def test_loader_strategy_with_missing_params_attribute(self):
        """Test loader strategy with config that doesn't have params attribute."""
        strategy = Crawl4AIWebLoaderStrategy()
        
        # Create a config object without params attribute
        class ConfigWithoutParams:
            def __init__(self):
                self.type = LoaderType.CRAWL4AI_WEB
                # No params attribute
        
        config = ConfigWithoutParams()
        url = "https://example.com"
        
        with patch('src.paas_ai.core.rag.loaders.crawl4ai_web.Crawl4AIWebLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            
            result = strategy.create_loader(config, url)
            
            # Should use empty dict as fallback since config has no params attribute
            mock_loader_class.assert_called_once_with(
                web_paths=[url],
                headless=True,
                wait_time=3.0,
                timeout=30000,
                enable_stealth=False
            )
    
    def test_loader_with_very_long_url(self):
        """Test loader with very long URL."""
        long_url = "https://example.com/" + "a" * 2000
        loader = Crawl4AIWebLoader(web_paths=[long_url])
        
        assert loader.web_paths == [long_url]
    
    def test_loader_with_special_characters_in_url(self):
        """Test loader with special characters in URL."""
        special_url = "https://example.com/path?param=value&other=test#fragment"
        loader = Crawl4AIWebLoader(web_paths=[special_url])
        
        assert loader.web_paths == [special_url]
    
    def test_loader_with_unicode_in_url(self):
        """Test loader with unicode characters in URL."""
        unicode_url = "https://example.com/测试"
        loader = Crawl4AIWebLoader(web_paths=[unicode_url])
        
        assert loader.web_paths == [unicode_url]
    
    def test_csv_parsing_with_unicode_content(self):
        """Test CSV parsing with unicode content."""
        csv_content = "url,title\nhttps://example.com,测试网站\nhttps://test.com,Another 测试"
        
        with temp_csv_file(csv_content) as csv_path:
            loader = Crawl4AIWebLoader()
            urls = loader._parse_csv_urls(str(csv_path))
            
            assert urls == ["https://example.com", "https://test.com"]
    
    def test_csv_parsing_with_quoted_urls(self):
        """Test CSV parsing with quoted URLs."""
        csv_content = 'url,title\n"https://example.com","Example Site"\n"https://test.com","Test Site"'
        
        with temp_csv_file(csv_content) as csv_path:
            loader = Crawl4AIWebLoader()
            urls = loader._parse_csv_urls(str(csv_path))
            
            assert urls == ["https://example.com", "https://test.com"]
    
    def test_csv_parsing_with_extra_whitespace(self):
        """Test CSV parsing with extra whitespace."""
        csv_content = " url , title \n https://example.com , Example \n https://test.com , Test "
        
        with temp_csv_file(csv_content) as csv_path:
            loader = Crawl4AIWebLoader()
            urls = loader._parse_csv_urls(str(csv_path))
            
            assert urls == ["https://example.com", "https://test.com"]
    
    def test_title_extraction_with_whitespace(self):
        """Test title extraction with extra whitespace."""
        loader = Crawl4AIWebLoader()
        content = "  #  Test Title  \n\nSome content."
        
        title = loader._extract_title_from_markdown(content)
        assert title == "Test Title"
    
    def test_title_extraction_with_multiple_spaces(self):
        """Test title extraction with multiple spaces in title."""
        loader = Crawl4AIWebLoader()
        content = "# Test   Title   With   Spaces\n\nSome content."
        
        title = loader._extract_title_from_markdown(content)
        assert title == "Test   Title   With   Spaces"
    
    def test_title_extraction_with_special_characters(self):
        """Test title extraction with special characters."""
        loader = Crawl4AIWebLoader()
        content = "# Test Title with Special Chars: @#$%^&*()\n\nSome content."
        
        title = loader._extract_title_from_markdown(content)
        assert title == "Test Title with Special Chars: @#$%^&*()"
    
    def test_title_extraction_with_unicode(self):
        """Test title extraction with unicode characters."""
        loader = Crawl4AIWebLoader()
        content = "# 测试标题 with Unicode\n\nSome content."
        
        title = loader._extract_title_from_markdown(content)
        assert title == "测试标题 with Unicode"

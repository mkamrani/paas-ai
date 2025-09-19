"""
Crawl4AI-based web loader for handling JavaScript-rendered content.

This loader uses Crawl4AI to properly handle SPAs like Docusaurus,
extracting clean markdown content from web pages.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path
import csv
import io

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.models import CrawlResult
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class Crawl4AIWebLoader(BaseLoader):
    """
    Web loader using Crawl4AI for JavaScript-rendered content.
    
    Features:
    - Handles JavaScript-rendered SPAs (like Docusaurus)
    - Batch processing of multiple URLs
    - Clean markdown extraction
    - Configurable browser and crawler settings
    """

    def __init__(
        self,
        web_paths: List[str] = None,
        web_path: str = None,
        headless: bool = True,
        wait_time: float = 3.0,
        timeout: int = 30000,
        enable_stealth: bool = True,
        **kwargs
    ):
        """
        Initialize the Crawl4AI web loader.
        
        Args:
            web_paths: List of URLs to crawl
            web_path: Single URL to crawl (alternative to web_paths)
            headless: Run browser in headless mode
            wait_time: Time to wait for page load
            timeout: Page timeout in milliseconds
            enable_stealth: Enable stealth mode for bot detection
            **kwargs: Additional arguments passed to BaseLoader
        """
        super().__init__()
        
        # Handle web paths (compatible with LangChain interface)
        if web_paths:
            self.web_paths = web_paths
        elif web_path:
            self.web_paths = [web_path]
        else:
            self.web_paths = []
            
        self.headless = headless
        self.wait_time = wait_time
        self.timeout = timeout
        self.enable_stealth = enable_stealth
        
        # Configure browser settings
        self.browser_config = BrowserConfig(
            headless=self.headless,
            enable_stealth=self.enable_stealth,
            viewport_width=1920,
            viewport_height=1080,
            verbose=True
        )
        
        # Configure crawler settings
        self.crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Always get fresh content
            page_timeout=self.timeout,
            delay_before_return_html=self.wait_time,
            wait_for="css:body",  # Wait for basic page structure
            remove_overlay_elements=True,  # Remove popups/modals
            verbose=True
        )

    async def _load_single_url(self, url: str) -> Document:
        """Load content from a single URL using Crawl4AI."""
        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                logger.info(f"Loading URL: {url}")
                
                result: CrawlResult = await crawler.arun(
                    url=url,
                    config=self.crawler_config
                )
                
                if not result.success:
                    raise Exception(f"Failed to crawl {url}: {result.error_message}")
                
                # Extract clean markdown content
                content = result.markdown or ""
                
                if not content.strip():
                    logger.warning(f"No content extracted from {url}")
                    content = "No content extracted"
                
                # Create document metadata
                metadata = {
                    "source": url,
                    "url": url,
                    "title": self._extract_title_from_markdown(content),
                    "content_length": len(content),
                    "success": True,
                    "loader_type": "crawl4ai_web"
                }
                
                logger.info(f"Successfully loaded {url} - {len(content)} characters")
                
                return Document(
                    page_content=content,
                    metadata=metadata
                )
                
        except Exception as e:
            logger.error(f"Error loading URL {url}: {str(e)}")
            return Document(
                page_content=f"Error loading {url}: {str(e)}",
                metadata={
                    "source": url,
                    "url": url,
                    "title": f"Error: {url}",
                    "content_length": 0,
                    "success": False,
                    "error": str(e),
                    "loader_type": "crawl4ai_web"
                }
            )

    async def _load_batch_urls(self, urls: List[str]) -> List[Document]:
        """Load content from multiple URLs using Crawl4AI batch processing."""
        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                logger.info(f"Loading {len(urls)} URLs in batch")
                
                # Use arun_many for concurrent processing
                results: List[CrawlResult] = await crawler.arun_many(
                    urls=urls,
                    config=self.crawler_config
                )
                
                documents = []
                
                for result in results:
                    try:
                        if result.success:
                            content = result.markdown or ""
                            
                            if not content.strip():
                                logger.warning(f"No content extracted from {result.url}")
                                content = "No content extracted"
                            
                            metadata = {
                                "source": result.url,
                                "url": result.url,
                                "title": self._extract_title_from_markdown(content),
                                "content_length": len(content),
                                "success": True,
                                "loader_type": "crawl4ai_web"
                            }
                            
                            logger.info(f"Successfully loaded {result.url} - {len(content)} characters")
                            
                        else:
                            content = f"Error loading {result.url}: {result.error_message}"
                            metadata = {
                                "source": result.url,
                                "url": result.url,
                                "title": f"Error: {result.url}",
                                "content_length": 0,
                                "success": False,
                                "error": result.error_message or "Unknown error",
                                "loader_type": "crawl4ai_web"
                            }
                            
                            logger.error(f"Failed to load {result.url}: {result.error_message}")
                        
                        documents.append(Document(
                            page_content=content,
                            metadata=metadata
                        ))
                        
                    except Exception as e:
                        logger.error(f"Error processing result for {result.url}: {str(e)}")
                        documents.append(Document(
                            page_content=f"Error processing {result.url}: {str(e)}",
                            metadata={
                                "source": result.url,
                                "url": result.url,
                                "title": f"Error: {result.url}",
                                "content_length": 0,
                                "success": False,
                                "error": str(e),
                                "loader_type": "crawl4ai_web"
                            }
                        ))
                
                return documents
                
        except Exception as e:
            logger.error(f"Error in batch URL loading: {str(e)}")
            # Return error documents for all URLs
            return [
                Document(
                    page_content=f"Batch error for {url}: {str(e)}",
                    metadata={
                        "source": url,
                        "url": url,
                        "title": f"Batch Error: {url}",
                        "content_length": 0,
                        "success": False,
                        "error": str(e),
                        "loader_type": "crawl4ai_web"
                    }
                )
                for url in urls
            ]

    def _extract_title_from_markdown(self, content: str) -> str:
        """Extract title from markdown content."""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return "Unknown Title"

    def load(self) -> List[Document]:
        """
        Load content from web URLs (LangChain interface).
        """
        try:
            # Check if there's already a running event loop
            loop = asyncio.get_running_loop()
            # If we're in an event loop, we need to use a different approach
            if loop.is_running():
                # Create a new event loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.aload())
                    return future.result()
            else:
                return asyncio.run(self.aload())
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.aload())
    
    async def aload(self) -> List[Document]:
        """
        Async load content from web URLs.
        """
        if not self.web_paths:
            logger.warning("No web paths specified")
            return []
        
        documents = []
        
        # Process URLs
        for web_path in self.web_paths:
            if web_path.endswith('.csv'):
                # CSV file with multiple URLs
                urls = self._parse_csv_urls(web_path)
                if not urls:
                    logger.warning(f"No URLs found in CSV file: {web_path}")
                    continue
                    
                logger.info(f"Loading {len(urls)} URLs from CSV: {web_path}")
                
                # Process in batches for better performance
                batch_size = 10  # Default batch size
                
                for i in range(0, len(urls), batch_size):
                    batch_urls = urls[i:i + batch_size]
                    batch_documents = await self._load_batch_urls(batch_urls)
                    documents.extend(batch_documents)
                    
            elif web_path.startswith(('http://', 'https://')):
                # Single URL
                document = await self._load_single_url(web_path)
                documents.append(document)
            else:
                logger.error(f"Unsupported source format: {web_path}")
                documents.append(Document(
                    page_content=f"Unsupported source format: {web_path}",
                    metadata={
                        "source": web_path,
                        "url": web_path,
                        "title": f"Error: {web_path}",
                        "content_length": 0,
                        "success": False,
                        "error": "Unsupported source format",
                        "loader_type": "crawl4ai_web"
                    }
                ))
        
        return documents

    def _parse_csv_urls(self, csv_path: str) -> List[str]:
        """Parse URLs from CSV file."""
        urls = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Try to detect CSV format
                sample = f.read(1024)
                f.seek(0)
                
                # Check if it has headers
                sniffer = csv.Sniffer()
                has_header = sniffer.has_header(sample)
                
                reader = csv.DictReader(f) if has_header else csv.reader(f)
                
                if has_header:
                    # Look for URL column
                    url_column = None
                    for fieldname in reader.fieldnames:
                        if fieldname and 'url' in fieldname.lower():
                            url_column = fieldname
                            break
                    
                    if not url_column:
                        raise ValueError("No URL column found in CSV headers")
                    
                    for row in reader:
                        url = row.get(url_column, '').strip()
                        if url and url.startswith(('http://', 'https://')):
                            urls.append(url)
                else:
                    # Assume first column contains URLs
                    for row in reader:
                        if row and len(row) > 0:
                            url = row[0].strip()
                            if url and url.startswith(('http://', 'https://')):
                                urls.append(url)
                                
        except Exception as e:
            logger.error(f"Error parsing CSV file {csv_path}: {str(e)}")
            raise
        
        logger.info(f"Parsed {len(urls)} URLs from CSV file")
        return urls

 
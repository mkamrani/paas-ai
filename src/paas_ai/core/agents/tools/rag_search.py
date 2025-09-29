"""
RAG search tool for agent integration.
"""

from typing import Any, Dict, Optional

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from paas_ai.utils.logging import get_logger

from ...config import Config, load_config
from ...rag import RAGProcessor

logger = get_logger("paas_ai.agents.tools.rag_search")


class RAGSearchInput(BaseModel):
    """Input schema for RAG search tool."""

    query: str = Field(description="The search query to find relevant information")
    limit: int = Field(default=5, description="Maximum number of results to return")


class RAGSearchTool(BaseTool):
    """Tool for searching the RAG knowledge base."""

    name: str = "search_knowledge_base"
    description: str = """
    Search the knowledge base for relevant information.
    Use this tool to find information about DSL specifications, documentation, or any indexed content.
    Provide a clear, specific query to get the best results.
    """
    args_schema: type = RAGSearchInput

    def __init__(self):
        """Initialize the RAG search tool without config dependency."""
        super().__init__()

    def _run(
        self,
        query: str,
        limit: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs,
    ) -> str:
        """Execute the search using config from runtime."""
        try:
            # logger.info(f"Running RAG Search Tool with query: {query} and limit: {limit}")
            # Get config from runtime or fallback to default
            config = None
            if run_manager and hasattr(run_manager, "config"):
                # Extract config from runtime
                configurable = getattr(run_manager.config, "configurable", {})
                config = configurable.get("paas_config")

            if not config:
                # Fallback to loading default config
                config = load_config()

            # logger.info(f"Config: {config}")

            # Create RAG processor with runtime config
            rag_processor = RAGProcessor(config)

            # Search the knowledge base
            results = rag_processor.search(query=query, limit=limit, include_metadata=True)

            logger.debug(f"Rag Processor Raw Results: {results}")

            if not results:
                return f"No information found for query: '{query}'"

            # Format results for the agent
            formatted_results = []
            for i, result in enumerate(results, 1):
                content = result["content"]
                score = result.get("score", 0.0)
                source = result.get("metadata", {}).get("source_url", "Unknown")

                # Build result string with citation if available
                result_parts = [f"Result {i} (score: {score:.2f}):", f"Content: {content}"]

                # Add citation information if available
                citation_info = result.get("citation")
                if citation_info:
                    formatted_citation = citation_info.get("formatted", "")
                    if formatted_citation:
                        result_parts.insert(1, f"Citation: {formatted_citation}")

                        # Add link if available
                        citation_link = citation_info.get("link")
                        if citation_link:
                            result_parts.append(f"Link: {citation_link}")
                    else:
                        # Fallback to basic source
                        result_parts.insert(1, f"Source: {source}")
                else:
                    # No citation available, use basic source
                    result_parts.insert(1, f"Source: {source}")

                formatted_results.append("\n".join(result_parts))

            return "\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return f"Error searching knowledge base: {str(e)}"

    async def _arun(self, query: str, limit: int = 5, **kwargs) -> str:
        """Async version of the search."""
        # For now, just call the sync version
        # In the future, we could make RAGProcessor.search async
        return self._run(query, limit, **kwargs)

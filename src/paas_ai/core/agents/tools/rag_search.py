"""
RAG search tool for agent integration.
"""

from typing import Any, Dict, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ...rag import RAGProcessor
from ...config import Config


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
    
    rag_processor: RAGProcessor = Field(exclude=True)
    
    def __init__(self, config: Config):
        """Initialize the RAG search tool."""
        super().__init__(rag_processor=RAGProcessor(config))
    
    def _run(self, query: str, limit: int = 5, **kwargs) -> str:
        """Execute the search and return formatted results."""
        try:
            # Search the knowledge base
            results = self.rag_processor.search(
                query=query,
                limit=limit,
                include_metadata=True
            )
            
            if not results:
                return f"No information found for query: '{query}'"
            
            # Format results for the agent
            formatted_results = []
            for i, result in enumerate(results, 1):
                content = result['content']
                score = result.get('score', 0.0)
                source = result.get('metadata', {}).get('source_url', 'Unknown')
                
                formatted_results.append(
                    f"Result {i} (score: {score:.2f}):\n"
                    f"Source: {source}\n"
                    f"Content: {content}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"
    
    async def _arun(self, query: str, limit: int = 5, **kwargs) -> str:
        """Async version of the search."""
        # For now, just call the sync version
        # In the future, we could make RAGProcessor.search async
        return self._run(query, limit, **kwargs) 
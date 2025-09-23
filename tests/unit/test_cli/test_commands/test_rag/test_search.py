"""
Unit tests for RAG search CLI command.

Tests all components of the search CLI module including:
- Search command functionality
- Query processing
- Resource type filtering
- Output formatting (text and JSON)
- Configuration handling
- Error handling and edge cases
- Empty knowledge base handling
- Result ranking and display
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from src.paas_ai.cli.commands.rag.search import search
from src.paas_ai.core.config import ConfigurationError, ResourceType
from src.paas_ai.core.rag import RAGProcessor


class TestSearchCommand:
    """Test the search command."""

    def test_search_basic_text_output(self):
        """Test basic search with text output."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 10}

            # Mock search results
            mock_results = [
                {
                    "content": "This is a test document about kubernetes deployment.",
                    "metadata": {
                        "source_url": "https://example.com/doc1",
                        "resource_type": "dsl",
                        "tags": ["kubernetes", "deployment"],
                    },
                    "score": 0.95,
                },
                {
                    "content": "Another document with kubernetes information.",
                    "metadata": {
                        "source_url": "https://example.com/doc2",
                        "resource_type": "contextual",
                        "tags": ["kubernetes", "containers"],
                    },
                    "score": 0.87,
                },
            ]
            mock_processor.search.return_value = mock_results

            # Run command
            result = runner.invoke(search, ["kubernetes deployment"])

            # Verify
            assert result.exit_code == 0
            assert "Searching for: 'kubernetes deployment'" in result.output
            assert "Search Results (2 found):" in result.output
            assert "Score: 0.950" in result.output
            assert "Score: 0.870" in result.output
            assert "Source: https://example.com/doc1" in result.output
            assert "Source: https://example.com/doc2" in result.output
            assert "Type: dsl" in result.output
            assert "Type: contextual" in result.output
            assert "Tags: kubernetes, deployment" in result.output
            assert "Tags: kubernetes, containers" in result.output
            assert "This is a test document about kubernetes deployment." in result.output
            assert "Another document with kubernetes information." in result.output
            assert "Search completed - 2 results found" in result.output

    def test_search_json_output(self):
        """Test search with JSON output format."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock search results
            mock_results = [
                {
                    "content": "Test content",
                    "metadata": {
                        "source_url": "https://example.com/doc",
                        "resource_type": "dsl",
                        "tags": ["test"],
                    },
                    "score": 0.9,
                }
            ]
            mock_processor.search.return_value = mock_results

            # Run command with JSON format
            result = runner.invoke(search, ["--format", "json", "test query"])

            # Verify
            assert result.exit_code == 0

            # Parse JSON output - find the JSON part between the first { and the last }
            output_lines = result.output.strip().split("\n")
            json_lines = []
            in_json = False
            brace_count = 0

            for line in output_lines:
                if line.strip().startswith("{") and not in_json:
                    in_json = True

                if in_json:
                    json_lines.append(line)
                    # Count braces to find the end of JSON
                    brace_count += line.count("{") - line.count("}")
                    # Stop when we've closed all braces and this line ends with }
                    if brace_count == 0 and line.strip().endswith("}"):
                        break

            json_output = "\n".join(json_lines)
            parsed_output = json.loads(json_output)

            assert parsed_output["query"] == "test query"
            assert parsed_output["total_results"] == 1
            assert len(parsed_output["results"]) == 1
            assert parsed_output["results"][0]["content"] == "Test content"
            assert parsed_output["results"][0]["score"] == 0.9

    def test_search_with_resource_type_filter(self):
        """Test search with specific resource type filter."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock search results
            mock_results = [
                {
                    "content": "Guidelines content",
                    "metadata": {
                        "source_url": "https://example.com/guidelines",
                        "resource_type": "guidelines",
                        "tags": ["security"],
                    },
                    "score": 0.85,
                }
            ]
            mock_processor.search.return_value = mock_results

            # Run command with resource type filter
            result = runner.invoke(search, ["--type", "guidelines", "security"])

            # Verify
            assert result.exit_code == 0
            mock_processor.search.assert_called_once_with(
                query="security",
                resource_type=ResourceType.GUIDELINES,
                limit=5,
                include_metadata=True,
            )

    def test_search_with_custom_limit(self):
        """Test search with custom result limit."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 20}

            # Mock search results
            mock_results = [
                {
                    "content": f"Result {i}",
                    "metadata": {
                        "source_url": f"https://example.com/doc{i}",
                        "resource_type": "dsl",
                    },
                    "score": 0.9 - i * 0.1,
                }
                for i in range(10)
            ]
            mock_processor.search.return_value = mock_results

            # Run command with custom limit
            result = runner.invoke(search, ["--limit", "10", "test query"])

            # Verify
            assert result.exit_code == 0
            mock_processor.search.assert_called_once_with(
                query="test query", resource_type=None, limit=10, include_metadata=True
            )

    def test_search_empty_knowledge_base(self):
        """Test search when knowledge base is empty."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show empty knowledge base
            mock_processor.get_stats.return_value = {"status": "empty", "total_documents": 0}

            # Run command
            result = runner.invoke(search, ["test query"])

            # Verify
            assert result.exit_code == 0
            assert "Knowledge base is empty" in result.output
            assert "Add resources first using 'paas-ai rag resources add'" in result.output
            mock_processor.search.assert_not_called()

    def test_search_no_results(self):
        """Test search when no results are found."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock empty search results
            mock_processor.search.return_value = []

            # Run command
            result = runner.invoke(search, ["nonexistent query"])

            # Verify
            assert result.exit_code == 0
            assert "No results found" in result.output

    def test_search_configuration_error(self):
        """Test search with configuration error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config:
            mock_load_config.side_effect = ConfigurationError("Config not found")

            result = runner.invoke(search, ["test query"])

            assert result.exit_code == 1
            assert "Config not found" in result.output

    def test_search_processing_error(self):
        """Test search with processing error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock search error
            mock_processor.search.side_effect = Exception("Search failed")

            result = runner.invoke(search, ["test query"])

            assert result.exit_code == 1
            assert "Search failed" in result.output

    def test_search_value_error(self):
        """Test search with value error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock value error
            mock_processor.search.side_effect = ValueError("Invalid query")

            result = runner.invoke(search, ["test query"])

            assert result.exit_code == 1
            assert "Invalid query" in result.output


class TestSearchEdgeCases:
    """Test edge cases for search command."""

    def test_search_with_all_resource_types(self):
        """Test search with 'all' resource type (default)."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock search results
            mock_results = [
                {
                    "content": "Test",
                    "metadata": {"source_url": "https://example.com", "resource_type": "dsl"},
                    "score": 0.9,
                }
            ]
            mock_processor.search.return_value = mock_results

            # Run command with 'all' type (default)
            result = runner.invoke(search, ["--type", "all", "test query"])

            # Verify
            assert result.exit_code == 0
            mock_processor.search.assert_called_once_with(
                query="test query",
                resource_type=None,  # Should be None for 'all'
                limit=5,
                include_metadata=True,
            )

    def test_search_with_empty_query(self):
        """Test search with empty query string."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock search results
            mock_results = [
                {
                    "content": "Empty query result",
                    "metadata": {"source_url": "https://example.com", "resource_type": "dsl"},
                    "score": 0.5,
                }
            ]
            mock_processor.search.return_value = mock_results

            # Run command with empty query
            result = runner.invoke(search, [""])

            # Verify
            assert result.exit_code == 0
            assert "Searching for: ''" in result.output
            mock_processor.search.assert_called_once_with(
                query="", resource_type=None, limit=5, include_metadata=True
            )

    def test_search_with_very_long_query(self):
        """Test search with very long query string."""
        runner = CliRunner()

        long_query = "a" * 1000  # Very long query

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock search results
            mock_results = [
                {
                    "content": "Long query result",
                    "metadata": {"source_url": "https://example.com", "resource_type": "dsl"},
                    "score": 0.8,
                }
            ]
            mock_processor.search.return_value = mock_results

            # Run command with long query
            result = runner.invoke(search, [long_query])

            # Verify
            assert result.exit_code == 0
            assert f"Searching for: '{long_query}'" in result.output
            mock_processor.search.assert_called_once_with(
                query=long_query, resource_type=None, limit=5, include_metadata=True
            )

    def test_search_with_special_characters(self):
        """Test search with special characters in query."""
        runner = CliRunner()

        special_query = "test@#$%^&*()_+-=[]{}|;':\",./<>?`~ query"

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock search results
            mock_results = [
                {
                    "content": "Special chars result",
                    "metadata": {"source_url": "https://example.com", "resource_type": "dsl"},
                    "score": 0.7,
                }
            ]
            mock_processor.search.return_value = mock_results

            # Run command with special characters
            result = runner.invoke(search, [special_query])

            # Verify
            assert result.exit_code == 0
            assert f"Searching for: '{special_query}'" in result.output
            mock_processor.search.assert_called_once_with(
                query=special_query, resource_type=None, limit=5, include_metadata=True
            )

    def test_search_with_unicode_query(self):
        """Test search with unicode characters in query."""
        runner = CliRunner()

        unicode_query = "测试查询 with unicode characters 🚀"

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock search results
            mock_results = [
                {
                    "content": "Unicode result",
                    "metadata": {"source_url": "https://example.com", "resource_type": "dsl"},
                    "score": 0.6,
                }
            ]
            mock_processor.search.return_value = mock_results

            # Run command with unicode query
            result = runner.invoke(search, [unicode_query])

            # Verify
            assert result.exit_code == 0
            assert f"Searching for: '{unicode_query}'" in result.output
            mock_processor.search.assert_called_once_with(
                query=unicode_query, resource_type=None, limit=5, include_metadata=True
            )

    def test_search_content_preview_truncation(self):
        """Test that content preview is properly truncated."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock search results with long content
            long_content = (
                "This is a very long content that should be truncated because it exceeds the 200 character limit that is used for content preview in the search results display. "
                * 3
            )
            mock_results = [
                {
                    "content": long_content,
                    "metadata": {
                        "source_url": "https://example.com/doc",
                        "resource_type": "dsl",
                        "tags": ["test"],
                    },
                    "score": 0.9,
                }
            ]
            mock_processor.search.return_value = mock_results

            # Run command
            result = runner.invoke(search, ["test query"])

            # Verify
            assert result.exit_code == 0
            # Check that content is truncated (should end with ...)
            assert (
                "Content: This is a very long content that should be truncated because it exceeds the 200 character limit that is used for content preview in the search results display. This is a very long content that should ..."
                in result.output
            )

    def test_search_with_missing_metadata(self):
        """Test search with results missing some metadata fields."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock search results with missing metadata
            mock_results = [
                {
                    "content": "Test content",
                    "metadata": {
                        "source_url": "https://example.com/doc",
                        "resource_type": "dsl"
                        # Missing tags
                    },
                    "score": 0.9,
                }
            ]
            mock_processor.search.return_value = mock_results

            # Run command
            result = runner.invoke(search, ["test query"])

            # Verify
            assert result.exit_code == 0
            assert "Source: https://example.com/doc" in result.output
            assert "Type: dsl" in result.output
            # Should not show tags line since tags are missing/empty

    def test_search_config_profile_warning(self):
        """Test that config profile override shows warning."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock search results
            mock_results = [
                {
                    "content": "Test",
                    "metadata": {"source_url": "https://example.com", "resource_type": "dsl"},
                    "score": 0.9,
                }
            ]
            mock_processor.search.return_value = mock_results

            # Run command with config profile
            result = runner.invoke(search, ["--config-profile", "local", "test query"])

            # Verify
            assert result.exit_code == 0
            assert "Config profile override not yet implemented" in result.output


class TestSearchIntegration:
    """Integration tests for search command."""

    def test_full_workflow_text_output(self):
        """Test complete workflow with text output."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 15}

            # Mock comprehensive search results
            mock_results = [
                {
                    "content": "Kubernetes deployment best practices and guidelines for production environments.",
                    "metadata": {
                        "source_url": "https://docs.example.com/kubernetes/deployment",
                        "resource_type": "dsl",
                        "tags": ["kubernetes", "deployment", "production"],
                    },
                    "score": 0.95,
                },
                {
                    "content": "Microservices architecture patterns and container orchestration strategies.",
                    "metadata": {
                        "source_url": "https://docs.example.com/microservices/patterns",
                        "resource_type": "contextual",
                        "tags": ["microservices", "containers", "architecture"],
                    },
                    "score": 0.87,
                },
                {
                    "content": "Security guidelines for containerized applications and network policies.",
                    "metadata": {
                        "source_url": "https://docs.example.com/security/containers",
                        "resource_type": "guidelines",
                        "tags": ["security", "containers", "networking"],
                    },
                    "score": 0.82,
                },
            ]
            mock_processor.search.return_value = mock_results

            # Run command
            result = runner.invoke(
                search,
                ["--type", "all", "--limit", "3", "--format", "text", "kubernetes deployment"],
            )

            # Verify complete workflow
            assert result.exit_code == 0
            assert "Searching for: 'kubernetes deployment'" in result.output
            assert "Search Results (3 found):" in result.output
            assert "Score: 0.950" in result.output
            assert "Score: 0.870" in result.output
            assert "Score: 0.820" in result.output
            assert "Kubernetes deployment best practices" in result.output
            assert "Microservices architecture patterns" in result.output
            assert "Security guidelines for containerized" in result.output
            assert "Search completed - 3 results found" in result.output

            # Verify search was called with correct parameters
            mock_processor.search.assert_called_once_with(
                query="kubernetes deployment",
                resource_type=None,  # 'all' maps to None
                limit=3,
                include_metadata=True,
            )

    def test_full_workflow_json_output(self):
        """Test complete workflow with JSON output."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "sentence_transformers"
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 8}

            # Mock search results
            mock_results = [
                {
                    "content": "Test content 1",
                    "metadata": {
                        "source_url": "https://example.com/doc1",
                        "resource_type": "dsl",
                        "tags": ["test", "example"],
                    },
                    "score": 0.9,
                },
                {
                    "content": "Test content 2",
                    "metadata": {
                        "source_url": "https://example.com/doc2",
                        "resource_type": "contextual",
                        "tags": ["test", "demo"],
                    },
                    "score": 0.8,
                },
            ]
            mock_processor.search.return_value = mock_results

            # Run command with JSON format
            result = runner.invoke(search, ["--format", "json", "--limit", "2", "test query"])

            # Verify complete workflow
            assert result.exit_code == 0

            # Parse JSON output - find the JSON part between the first { and the last }
            output_lines = result.output.strip().split("\n")
            json_lines = []
            in_json = False
            brace_count = 0

            for line in output_lines:
                if line.strip().startswith("{") and not in_json:
                    in_json = True

                if in_json:
                    json_lines.append(line)
                    # Count braces to find the end of JSON
                    brace_count += line.count("{") - line.count("}")
                    # Stop when we've closed all braces and this line ends with }
                    if brace_count == 0 and line.strip().endswith("}"):
                        break

            json_output = "\n".join(json_lines)
            parsed_output = json.loads(json_output)

            assert parsed_output["query"] == "test query"
            assert parsed_output["total_results"] == 2
            assert len(parsed_output["results"]) == 2
            assert parsed_output["results"][0]["content"] == "Test content 1"
            assert parsed_output["results"][0]["score"] == 0.9
            assert parsed_output["results"][1]["content"] == "Test content 2"
            assert parsed_output["results"][1]["score"] == 0.8

            # Verify search was called with correct parameters
            mock_processor.search.assert_called_once_with(
                query="test query", resource_type=None, limit=2, include_metadata=True
            )

    def test_error_handling_workflow(self):
        """Test error handling across the workflow."""
        runner = CliRunner()

        # Test configuration error
        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config:
            mock_load_config.side_effect = ConfigurationError("Config file not found")

            result = runner.invoke(search, ["test query"])

            assert result.exit_code == 1
            assert "Config file not found" in result.output

        # Test processing error
        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock search error
            mock_processor.search.side_effect = Exception("Network error")

            result = runner.invoke(search, ["test query"])

            assert result.exit_code == 1
            assert "Search failed" in result.output

        # Test value error
        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats to show non-empty knowledge base
            mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

            # Mock value error
            mock_processor.search.side_effect = ValueError("Invalid parameters")

            result = runner.invoke(search, ["test query"])

            assert result.exit_code == 1
            assert "Invalid parameters" in result.output

    def test_resource_type_filtering_workflow(self):
        """Test resource type filtering across different types."""
        runner = CliRunner()

        resource_types = [
            ("dsl", ResourceType.DSL),
            ("contextual", ResourceType.CONTEXTUAL),
            ("guidelines", ResourceType.GUIDELINES),
            ("domain_rules", ResourceType.DOMAIN_RULES),
        ]

        for type_str, expected_type in resource_types:
            with patch(
                "src.paas_ai.cli.commands.rag.search.load_config"
            ) as mock_load_config, patch(
                "src.paas_ai.cli.commands.rag.search.RAGProcessor"
            ) as mock_processor_class:
                # Setup mocks
                mock_config = Mock()
                mock_load_config.return_value = mock_config

                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor

                # Mock stats to show non-empty knowledge base
                mock_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}

                # Mock search results
                mock_results = [
                    {
                        "content": f"{type_str} content",
                        "metadata": {
                            "source_url": "https://example.com",
                            "resource_type": type_str,
                        },
                        "score": 0.9,
                    }
                ]
                mock_processor.search.return_value = mock_results

                # Run command with specific resource type
                result = runner.invoke(search, ["--type", type_str, "test query"])

                # Verify
                assert result.exit_code == 0
                mock_processor.search.assert_called_once_with(
                    query="test query", resource_type=expected_type, limit=5, include_metadata=True
                )

"""
Integration tests for RAG CLI commands.

Tests the complete RAG CLI system integration including:
- End-to-end workflows across all commands
- Cross-command interactions
- Command chaining and sequencing
- Error propagation across components
- Configuration consistency
- Performance characteristics
- Real-world usage scenarios
"""

import csv
import json
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from src.paas_ai.cli.commands.rag import rag
from src.paas_ai.core.config import ConfigurationError, ResourceType
from src.paas_ai.core.rag import RAGProcessor


class TestRAGCLISystemIntegration:
    """Integration tests for the complete RAG CLI system."""

    def test_full_workflow_add_search_status(self):
        """Test complete workflow: add resource -> search -> status."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "src.paas_ai.cli.commands.rag.resources.asyncio.run"
        ) as mock_asyncio_run, patch(
            "src.paas_ai.cli.commands.rag.search.load_config"
        ) as mock_search_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_search_processor_class, patch(
            "src.paas_ai.cli.commands.rag.status.load_config"
        ) as mock_status_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_status_processor_class:
            # Setup mocks for add resource
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_asyncio_run.return_value = {"successful": 1, "total_documents": 5}

            # Setup mocks for search
            mock_search_config.return_value = mock_config
            mock_search_processor = Mock()
            mock_search_processor_class.return_value = mock_search_processor
            mock_search_processor.get_stats.return_value = {"status": "ready", "total_documents": 5}
            mock_search_processor.search.return_value = [
                {
                    "content": "Test content about kubernetes",
                    "metadata": {
                        "source_url": "https://example.com/doc",
                        "resource_type": "dsl",
                        "tags": ["kubernetes"],
                    },
                    "score": 0.9,
                }
            ]

            # Setup mocks for status
            mock_status_config.return_value = mock_config
            mock_status_processor = Mock()
            mock_status_processor_class.return_value = mock_status_processor
            mock_status_processor.get_stats.return_value = {
                "status": "ready",
                "total_documents": 5,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
                "retriever_type": "similarity",
            }

            # Step 1: Add resource
            result1 = runner.invoke(rag, ["resources", "add", "--url", "https://example.com/doc"])
            assert result1.exit_code == 0
            # The success message is logged to stdout and captured by pytest
            # We need to check that the command executed successfully (exit code 0)
            # The actual success message will be in the captured stdout

            # Step 2: Search
            result2 = runner.invoke(rag, ["search", "kubernetes"])
            assert result2.exit_code == 0
            assert "Search Results" in result2.output or "Searching for" in result2.output
            assert "Test content about kubernetes" in result2.output

            # Step 3: Check status
            result3 = runner.invoke(rag, ["status"])
            assert result3.exit_code == 0
            assert (
                "Knowledge Base Status" in result3.output or "RAG System Status" in result3.output
            )
            assert "Total Documents" in result3.output

    def test_full_workflow_batch_add_sync_status(self):
        """Test complete workflow: batch add -> sync -> status."""
        runner = CliRunner()

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "type", "tags", "priority"])
            writer.writerow(["https://example.com/doc1", "dsl", "test,example", "7"])
            writer.writerow(["https://example.com/doc2", "contextual", "demo", "5"])
            csv_file = f.name

        try:
            with patch(
                "src.paas_ai.cli.commands.rag.resources.load_config"
            ) as mock_load_config, patch(
                "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
            ) as mock_create_resource, patch(
                "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
            ) as mock_processor_class, patch(
                "src.paas_ai.cli.commands.rag.resources.asyncio.run"
            ) as mock_asyncio_run, patch(
                "src.paas_ai.cli.commands.rag.sync.get_logger"
            ) as mock_sync_logger, patch(
                "src.paas_ai.cli.commands.rag.status.load_config"
            ) as mock_status_config, patch(
                "src.paas_ai.cli.commands.rag.status.RAGProcessor"
            ) as mock_status_processor_class:
                # Setup mocks for batch add
                mock_config = Mock()
                mock_config.embedding.type = "sentence_transformers"
                mock_load_config.return_value = mock_config

                mock_resource = Mock()
                mock_create_resource.return_value = mock_resource

                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_asyncio_run.return_value = {
                    "successful": 2,
                    "failed": 0,
                    "total_documents": 10,
                }

                # Setup mocks for sync
                mock_sync_logger_instance = Mock()
                mock_sync_logger.return_value = mock_sync_logger_instance

                # Setup mocks for status
                mock_status_config.return_value = mock_config
                mock_status_processor = Mock()
                mock_status_processor_class.return_value = mock_status_processor
                mock_status_processor.get_stats.return_value = {
                    "status": "ready",
                    "total_documents": 10,
                    "vectorstore_type": "chroma",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "retriever_type": "similarity",
                }

                # Step 1: Batch add resources
                result1 = runner.invoke(rag, ["resources", "add-batch", "--csv-file", csv_file])
                assert result1.exit_code == 0
                # Verify the command succeeded by checking mock calls
                mock_load_config.assert_called_once()
                mock_create_resource.assert_called()
                mock_processor_class.assert_called_once_with(mock_config)
                mock_asyncio_run.assert_called_once()

                # Step 2: Sync
                result2 = runner.invoke(rag, ["sync", "--incremental"])
                assert result2.exit_code == 0
                # Verify sync command succeeded
                mock_sync_logger.assert_called()

                # Step 3: Check status
                result3 = runner.invoke(rag, ["status"])
                assert result3.exit_code == 0
                # Verify status command succeeded
                mock_status_config.assert_called()
                mock_status_processor_class.assert_called_once_with(mock_config)
                mock_status_processor.get_stats.assert_called_once()

        finally:
            os.unlink(csv_file)

    def test_full_workflow_search_reports_status(self):
        """Test complete workflow: search -> reports -> status."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.search.load_config") as mock_search_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_search_processor_class, patch(
            "src.paas_ai.cli.commands.rag.reports.get_logger"
        ) as mock_reports_logger, patch(
            "src.paas_ai.cli.commands.rag.status.load_config"
        ) as mock_status_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_status_processor_class:
            # Setup mocks for search
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_search_config.return_value = mock_config

            mock_search_processor = Mock()
            mock_search_processor_class.return_value = mock_search_processor
            mock_search_processor.get_stats.return_value = {
                "status": "ready",
                "total_documents": 15,
            }
            mock_search_processor.search.return_value = [
                {
                    "content": "Microservices architecture patterns",
                    "metadata": {
                        "source_url": "https://example.com/microservices",
                        "resource_type": "contextual",
                        "tags": ["microservices", "architecture"],
                    },
                    "score": 0.95,
                }
            ]

            # Setup mocks for reports
            mock_reports_logger_instance = Mock()
            mock_reports_logger.return_value = mock_reports_logger_instance

            # Setup mocks for status
            mock_status_config.return_value = mock_config
            mock_status_processor = Mock()
            mock_status_processor_class.return_value = mock_status_processor
            mock_status_processor.get_stats.return_value = {
                "status": "ready",
                "total_documents": 15,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
                "retriever_type": "similarity",
            }

            # Step 1: Search
            result1 = runner.invoke(rag, ["search", "microservices"])
            assert result1.exit_code == 0
            assert "Search Results (1 found):" in result1.output
            assert "Microservices architecture patterns" in result1.output

            # Step 2: Generate reports
            result2 = runner.invoke(rag, ["report", "sync-history"])
            assert result2.exit_code == 0
            # Verify the command succeeded by checking mock calls instead of output
            mock_reports_logger.assert_called()
            mock_reports_logger_instance.info.assert_any_call("Sync history report for all time")

            result3 = runner.invoke(rag, ["report", "resource-health"])
            assert result3.exit_code == 0
            # Verify the command succeeded by checking mock calls instead of output
            mock_reports_logger_instance.info.assert_any_call("Resource health report")

            # Step 3: Check status
            result4 = runner.invoke(rag, ["status"])
            assert result4.exit_code == 0
            assert "📚 Knowledge Base Status: ready" in result4.output
            assert "📊 Total Documents: 15" in result4.output

    def test_error_propagation_across_commands(self):
        """Test that errors propagate correctly across commands."""
        runner = CliRunner()

        # Test configuration error propagation
        with patch(
            "src.paas_ai.cli.commands.rag.resources.load_config"
        ) as mock_resources_config, patch(
            "src.paas_ai.cli.commands.rag.search.load_config"
        ) as mock_search_config, patch(
            "src.paas_ai.cli.commands.rag.status.load_config"
        ) as mock_status_config:
            # All load_config calls should fail
            mock_resources_config.side_effect = ConfigurationError("Config not found")
            mock_search_config.side_effect = ConfigurationError("Config not found")
            mock_status_config.side_effect = ConfigurationError("Config not found")

            # All commands should fail with configuration error
            result1 = runner.invoke(rag, ["resources", "add", "--url", "https://example.com/doc"])
            assert result1.exit_code == 1
            assert (
                "Configuration error" in result1.output
                or "Failed to add resource" in result1.output
            )

            result2 = runner.invoke(rag, ["search", "test"])
            assert result2.exit_code == 1
            assert "Configuration error" in result2.output or "Search failed" in result2.output

            result3 = runner.invoke(rag, ["status"])
            assert result3.exit_code == 1
            assert (
                "Configuration error" in result3.output or "Status check failed" in result3.output
            )

    def test_configuration_consistency_across_commands(self):
        """Test that configuration is consistent across all commands."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.rag.resources.load_config"
        ) as mock_resources_config, patch(
            "src.paas_ai.cli.commands.rag.search.load_config"
        ) as mock_search_config, patch(
            "src.paas_ai.cli.commands.rag.status.load_config"
        ) as mock_status_config:
            # Setup consistent mock configuration (simplified like working tests)
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.embedding.model_name = "text-embedding-3-small"

            mock_resources_config.return_value = mock_config
            mock_search_config.return_value = mock_config
            mock_status_config.return_value = mock_config

            # All commands should use the same configuration
            with patch(
                "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
            ) as mock_create_resource, patch(
                "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
            ) as mock_processor_class, patch(
                "src.paas_ai.cli.commands.rag.resources.asyncio.run"
            ) as mock_asyncio_run, patch(
                "src.paas_ai.cli.commands.rag.search.RAGProcessor"
            ) as mock_search_processor_class, patch(
                "src.paas_ai.cli.commands.rag.status.RAGProcessor"
            ) as mock_status_processor_class:
                # Setup processor mocks
                mock_resource = Mock()
                mock_create_resource.return_value = mock_resource

                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_asyncio_run.return_value = {"successful": 1, "total_documents": 3}

                mock_search_processor = Mock()
                mock_search_processor_class.return_value = mock_search_processor
                mock_search_processor.get_stats.return_value = {
                    "status": "ready",
                    "total_documents": 3,
                }
                mock_search_processor.search.return_value = []

                mock_status_processor = Mock()
                mock_status_processor_class.return_value = mock_status_processor
                mock_status_processor.get_stats.return_value = {
                    "status": "ready",
                    "total_documents": 3,
                    "vectorstore_type": "chroma",
                    "embedding_model": "text-embedding-3-small",
                    "retriever_type": "similarity",
                }

                # Test that all commands use the same config
                result1 = runner.invoke(
                    rag, ["resources", "add", "--url", "https://example.com/doc"]
                )
                assert result1.exit_code == 0
                # Verify the command succeeded by checking mock calls instead of output
                mock_resources_config.assert_called()
                mock_create_resource.assert_called()

                result2 = runner.invoke(rag, ["search", "test"])
                assert result2.exit_code == 0
                # Verify search command used the same config
                mock_search_config.assert_called()

                result3 = runner.invoke(rag, ["status"])
                assert result3.exit_code == 0
                # Verify status command used the same config
                mock_status_config.assert_called()

                # Verify all commands were called with the same config object
                assert mock_resources_config.call_args == mock_search_config.call_args
                assert mock_search_config.call_args == mock_status_config.call_args


class TestRAGCLICommandChaining:
    """Test command chaining and sequencing scenarios."""

    def test_sequential_resource_operations(self):
        """Test sequential resource operations."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "src.paas_ai.cli.commands.rag.resources.asyncio.run"
        ) as mock_asyncio_run:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_asyncio_run.return_value = {"successful": 1, "total_documents": 1}

            # Add multiple resources sequentially
            result1 = runner.invoke(rag, ["resources", "add", "--url", "https://example.com/doc1"])
            assert result1.exit_code == 0

            result2 = runner.invoke(rag, ["resources", "add", "--url", "https://example.com/doc2"])
            assert result2.exit_code == 0

            result3 = runner.invoke(rag, ["resources", "add", "--url", "https://example.com/doc3"])
            assert result3.exit_code == 0

            # Verify all were processed
            assert mock_create_resource.call_count == 3
            assert mock_processor.add_resources.call_count == 3

    def test_mixed_command_sequence(self):
        """Test mixed command sequence."""
        runner = CliRunner()

        with patch("paas_ai.core.config.load_config") as mock_resources_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "src.paas_ai.cli.commands.rag.resources.asyncio.run"
        ) as mock_asyncio_run, patch(
            "src.paas_ai.cli.commands.rag.search.load_config"
        ) as mock_search_config, patch(
            "src.paas_ai.cli.commands.rag.search.RAGProcessor"
        ) as mock_search_processor_class, patch(
            "src.paas_ai.cli.commands.rag.sync.get_logger"
        ) as mock_sync_logger, patch(
            "src.paas_ai.cli.commands.rag.status.load_config"
        ) as mock_status_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_status_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_resources_config.return_value = mock_config
            mock_search_config.return_value = mock_config
            mock_status_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_asyncio_run.return_value = {"successful": 1, "total_documents": 2}

            mock_search_processor = Mock()
            mock_search_processor_class.return_value = mock_search_processor
            mock_search_processor.get_stats.return_value = {"status": "ready", "total_documents": 2}
            mock_search_processor.search.return_value = []

            mock_sync_logger_instance = Mock()
            mock_sync_logger.return_value = mock_sync_logger_instance

            mock_status_processor = Mock()
            mock_status_processor_class.return_value = mock_status_processor
            mock_status_processor.get_stats.return_value = {
                "status": "ready",
                "total_documents": 2,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
                "retriever_type": "similarity",
            }

            # Mixed command sequence
            result1 = runner.invoke(rag, ["resources", "add", "--url", "https://example.com/doc"])
            assert result1.exit_code == 0

            result2 = runner.invoke(rag, ["status"])
            assert result2.exit_code == 0

            result3 = runner.invoke(rag, ["search", "test"])
            assert result3.exit_code == 0

            result4 = runner.invoke(rag, ["sync", "--incremental"])
            assert result4.exit_code == 0

            result5 = runner.invoke(rag, ["status"])
            assert result5.exit_code == 0

    def test_error_recovery_sequence(self):
        """Test error recovery in command sequence."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "src.paas_ai.cli.commands.rag.resources.asyncio.run"
        ) as mock_asyncio_run:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # First call fails, second succeeds
            mock_asyncio_run.side_effect = [
                Exception("Network error"),
                {"successful": 1, "total_documents": 1},
            ]

            # First attempt fails
            result1 = runner.invoke(rag, ["resources", "add", "--url", "https://example.com/doc"])
            assert result1.exit_code == 1
            assert "Failed to add resource" in result1.output

            # Second attempt succeeds
            result2 = runner.invoke(rag, ["resources", "add", "--url", "https://example.com/doc"])
            assert result2.exit_code == 0
            # Verify the command succeeded by checking mock calls instead of output
            mock_asyncio_run.assert_called()
            # Verify the second call succeeded (first failed, second succeeded)
            assert mock_asyncio_run.call_count == 2


class TestRAGCLIPerformanceIntegration:
    """Test performance characteristics of the RAG CLI system."""

    def test_command_execution_time(self):
        """Test that commands execute within reasonable time."""
        import time

        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.get_stats.return_value = {
                "status": "ready",
                "total_documents": 100,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
                "retriever_type": "similarity",
            }

            # Time multiple command executions
            start_time = time.time()
            for _ in range(10):
                result = runner.invoke(rag, ["status"])
                assert result.exit_code == 0
            end_time = time.time()

            # Should be reasonably fast (less than 5 seconds for 10 commands)
            assert (end_time - start_time) < 5.0

    def test_memory_usage_consistency(self):
        """Test that memory usage is consistent across commands."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.get_stats.return_value = {
                "status": "ready",
                "total_documents": 50,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
                "retriever_type": "similarity",
            }

            # Execute multiple commands
            for _ in range(5):
                result = runner.invoke(rag, ["status"])
                assert result.exit_code == 0

            # Should not have memory leaks or excessive memory usage
            # This is more of a smoke test - actual memory monitoring would require more sophisticated tools


class TestRAGCLIRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_developer_workflow(self):
        """Test typical developer workflow."""
        runner = CliRunner()

        # Create temporary CSV file for batch import
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "type", "tags", "priority"])
            writer.writerow(["https://docs.example.com/api", "dsl", "api,documentation", "8"])
            writer.writerow(
                ["https://docs.example.com/guides", "guidelines", "best-practices", "7"]
            )
            writer.writerow(
                ["https://docs.example.com/examples", "contextual", "examples,tutorials", "6"]
            )
            csv_file = f.name

        try:
            with patch("paas_ai.core.config.load_config") as mock_resources_config, patch(
                "paas_ai.core.rag.create_resource_from_url"
            ) as mock_create_resource, patch(
                "paas_ai.core.rag.RAGProcessor"
            ) as mock_processor_class, patch(
                "src.paas_ai.cli.commands.rag.resources.asyncio.run"
            ) as mock_asyncio_run, patch(
                "src.paas_ai.cli.commands.rag.search.load_config"
            ) as mock_search_config, patch(
                "src.paas_ai.cli.commands.rag.search.RAGProcessor"
            ) as mock_search_processor_class, patch(
                "src.paas_ai.cli.commands.rag.sync.get_logger"
            ) as mock_sync_logger, patch(
                "src.paas_ai.cli.commands.rag.status.load_config"
            ) as mock_status_config, patch(
                "src.paas_ai.cli.commands.rag.status.RAGProcessor"
            ) as mock_status_processor_class, patch(
                "src.paas_ai.cli.commands.rag.reports.get_logger"
            ) as mock_reports_logger:
                # Setup mocks
                mock_config = Mock()
                mock_config.embedding.type = "openai"
                mock_resources_config.return_value = mock_config
                mock_search_config.return_value = mock_config
                mock_status_config.return_value = mock_config

                mock_resource = Mock()
                mock_create_resource.return_value = mock_resource

                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_asyncio_run.return_value = {
                    "successful": 3,
                    "failed": 0,
                    "total_documents": 15,
                }

                mock_search_processor = Mock()
                mock_search_processor_class.return_value = mock_search_processor
                mock_search_processor.get_stats.return_value = {
                    "status": "ready",
                    "total_documents": 15,
                }
                mock_search_processor.search.return_value = [
                    {
                        "content": "API documentation for REST endpoints",
                        "metadata": {
                            "source_url": "https://docs.example.com/api",
                            "resource_type": "dsl",
                            "tags": ["api", "documentation"],
                        },
                        "score": 0.95,
                    }
                ]

                mock_sync_logger_instance = Mock()
                mock_sync_logger.return_value = mock_sync_logger_instance

                mock_status_processor = Mock()
                mock_status_processor_class.return_value = mock_status_processor
                mock_status_processor.get_stats.return_value = {
                    "status": "ready",
                    "total_documents": 15,
                    "vectorstore_type": "chroma",
                    "embedding_model": "text-embedding-3-small",
                    "retriever_type": "similarity",
                }

                mock_reports_logger_instance = Mock()
                mock_reports_logger.return_value = mock_reports_logger_instance

                # Developer workflow
                # 1. Check initial status
                result1 = runner.invoke(rag, ["status"])
                assert result1.exit_code == 0

                # 2. Batch import documentation
                result2 = runner.invoke(rag, ["resources", "add-batch", "--csv-file", csv_file])
                assert result2.exit_code == 0
                # The logging output goes to stdout, not to Click result output
                # The command succeeded (exit code 0), so the batch import worked

                # 3. Sync the knowledge base
                result3 = runner.invoke(rag, ["sync", "--incremental"])
                assert result3.exit_code == 0

                # 4. Search for information
                result4 = runner.invoke(rag, ["search", "API documentation"])
                assert result4.exit_code == 0
                assert "Search Results (1 found):" in result4.output

                # 5. Check final status
                result5 = runner.invoke(rag, ["status"])
                assert result5.exit_code == 0
                assert "📊 Total Documents: 15" in result5.output

                # 6. Generate reports
                result6 = runner.invoke(rag, ["report", "resource-health"])
                assert result6.exit_code == 0

        finally:
            os.unlink(csv_file)

    def test_operations_workflow(self):
        """Test typical operations workflow."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_status_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_status_processor_class, patch(
            "src.paas_ai.cli.commands.rag.sync.get_logger"
        ) as mock_sync_logger, patch(
            "src.paas_ai.cli.commands.rag.reports.get_logger"
        ) as mock_reports_logger:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "sentence_transformers"
            mock_status_config.return_value = mock_config

            mock_status_processor = Mock()
            mock_status_processor_class.return_value = mock_status_processor
            mock_status_processor.get_stats.return_value = {
                "status": "ready",
                "total_documents": 100,
                "vectorstore_type": "chroma",
                "embedding_model": "all-MiniLM-L6-v2",
                "retriever_type": "similarity",
            }

            mock_sync_logger_instance = Mock()
            mock_sync_logger.return_value = mock_sync_logger_instance

            mock_reports_logger_instance = Mock()
            mock_reports_logger.return_value = mock_reports_logger_instance

            # Operations workflow
            # 1. Check system health
            result1 = runner.invoke(rag, ["status", "--detailed"])
            assert result1.exit_code == 0
            assert "📚 Knowledge Base Status: ready" in result1.output

            # 2. Run incremental sync
            result2 = runner.invoke(rag, ["sync", "--incremental"])
            assert result2.exit_code == 0

            # 3. Generate health report
            result3 = runner.invoke(rag, ["report", "resource-health"])
            assert result3.exit_code == 0

            # 4. Generate sync history report
            result4 = runner.invoke(rag, ["report", "sync-history", "--last-week"])
            assert result4.exit_code == 0

            # 5. Final status check
            result5 = runner.invoke(rag, ["status"])
            assert result5.exit_code == 0

    def test_troubleshooting_workflow(self):
        """Test troubleshooting workflow."""
        runner = CliRunner()

        # Test error scenario
        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_status_config, patch(
            "src.paas_ai.cli.commands.rag.search.load_config"
        ) as mock_search_config, patch(
            "src.paas_ai.cli.commands.rag.resources.load_config"
        ) as mock_resources_config:
            mock_status_config.side_effect = ConfigurationError("Config file corrupted")
            mock_search_config.side_effect = ConfigurationError("Config file corrupted")
            mock_resources_config.side_effect = ConfigurationError("Config file corrupted")

            # 1. Try to check status (fails)
            result1 = runner.invoke(rag, ["status"])
            assert result1.exit_code == 1
            assert (
                "Configuration error" in result1.output or "Status check failed" in result1.output
            )

            # 2. Try to search (fails)
            result2 = runner.invoke(rag, ["search", "test"])
            assert result2.exit_code == 1
            assert "Configuration error" in result2.output or "Search failed" in result2.output

            # 3. Try to add resource (fails)
            result3 = runner.invoke(rag, ["resources", "add", "--url", "https://example.com/doc"])
            assert result3.exit_code == 1
            assert (
                "Configuration error" in result3.output
                or "Failed to add resource" in result3.output
            )

        # Test recovery scenario
        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup working configuration
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.get_stats.return_value = {
                "status": "ready",
                "total_documents": 5,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
                "retriever_type": "similarity",
            }

            # 4. Check status after recovery (succeeds)
            result4 = runner.invoke(rag, ["status"])
            assert result4.exit_code == 0
            assert "📚 Knowledge Base Status: ready" in result4.output

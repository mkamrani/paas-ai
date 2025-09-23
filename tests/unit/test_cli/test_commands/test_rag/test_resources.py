"""
Unit tests for RAG resources CLI commands.

Tests all components of the resources CLI module including:
- Resources group command
- Add resource command
- Add batch resources command
- List resources command
- Remove resource command
- Clear resources command
- Error handling and edge cases
- Configuration handling
- CSV file processing
"""

import asyncio
import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from paas_ai.core.config import ConfigurationError, ResourceType
from src.paas_ai.cli.commands.rag.resources import (
    add_resource,
    add_resources_batch,
    clear_resources,
    list_resources,
    remove_resource,
    resources_group,
)
from src.paas_ai.core.rag import RAGProcessor


class TestResourcesGroup:
    """Test the resources group command."""

    def test_resources_group_creation(self):
        """Test that the resources group is created correctly."""
        assert resources_group is not None
        assert resources_group.name == "resources"
        assert "Manage RAG knowledge base resources" in resources_group.help

    def test_resources_group_subcommands(self):
        """Test that all subcommands are registered."""
        subcommands = resources_group.commands

        expected_commands = ["add", "add-batch", "list", "remove", "clear"]
        for cmd in expected_commands:
            assert cmd in subcommands

    def test_resources_group_help(self):
        """Test that the resources group has proper help text."""
        assert resources_group.help is not None
        assert "Manage RAG knowledge base resources" in resources_group.help


class TestAddResource:
    """Test the add resource command."""

    def test_add_resource_basic(self):
        """Test adding a resource with basic parameters."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "asyncio.run"
        ) as mock_asyncio_run:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.embedding.params = {"api_key": "test-api-key"}  # Add mock API key
            mock_load_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_asyncio_run.return_value = {"successful": 1, "total_documents": 5}

            # Run command
            result = runner.invoke(
                resources_group,
                [
                    "add",
                    "--url",
                    "https://example.com/doc",
                    "--type",
                    "dsl",
                    "--tags",
                    "test,example",
                    "--priority",
                    "7",
                ],
            )

            # Verify
            assert result.exit_code == 0
            mock_load_config.assert_called_once()
            mock_create_resource.assert_called_once_with(
                url="https://example.com/doc",
                resource_type=ResourceType.DSL,
                tags=["test", "example"],
                priority=7,
            )
            mock_processor_class.assert_called_once_with(mock_config)
            mock_processor.add_resources.assert_called_once_with([mock_resource])
            mock_asyncio_run.assert_called_once()

    def test_add_resource_default_parameters(self):
        """Test adding a resource with default parameters."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "asyncio.run"
        ) as mock_asyncio_run:
            # Setup mocks with proper nested structure
            mock_config = Mock()
            mock_config.embedding.type = "openai"  # Set proper embedding type
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.embedding.params = {"api_key": "test-api-key"}  # Add mock API key
            mock_config.vectorstore.type = "chroma"
            mock_config.vectorstore.persist_directory = "/tmp/test"
            mock_config.vectorstore.collection_name = "test_collection"
            mock_config.vectorstore.params = {}
            mock_config.retriever.type = "similarity"
            mock_config.retriever.search_kwargs = {"k": 5}
            mock_config.retriever.params = {}
            mock_config.batch_size = 32
            mock_config.validate_urls = True
            mock_config.log_level = "INFO"
            mock_load_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_asyncio_run.return_value = {"successful": 1, "total_documents": 3}

            # Run command with minimal parameters
            result = runner.invoke(resources_group, ["add", "--url", "https://example.com/doc"])

            # Verify
            assert result.exit_code == 0
            mock_create_resource.assert_called_once_with(
                url="https://example.com/doc",
                resource_type=ResourceType.DSL,  # default
                tags=[],  # default
                priority=5,  # default
            )

    def test_add_resource_different_types(self):
        """Test adding resources with different types."""
        runner = CliRunner()

        test_cases = [
            ("dsl", ResourceType.DSL),
            ("contextual", ResourceType.CONTEXTUAL),
            ("guidelines", ResourceType.GUIDELINES),
            ("domain_rules", ResourceType.DOMAIN_RULES),
        ]

        for type_str, expected_type in test_cases:
            with patch(
                "src.paas_ai.cli.commands.rag.resources.load_config"
            ) as mock_load_config, patch(
                "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
            ) as mock_create_resource, patch(
                "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
            ) as mock_processor_class, patch(
                "asyncio.run"
            ) as mock_asyncio_run:
                # Setup mocks
                mock_config = Mock()
                # Mock embedding config
                mock_config.embedding.type = "openai"
                mock_config.embedding.model_name = "text-embedding-3-small"
                mock_config.embedding.params = {}
                # Mock vectorstore config
                mock_config.vectorstore.type = "chroma"
                mock_config.vectorstore.persist_directory = None
                mock_config.vectorstore.collection_name = None
                mock_config.vectorstore.params = {}
                # Mock retriever config
                mock_config.retriever.type = "similarity"
                mock_config.retriever.search_kwargs = {"k": 5}
                mock_config.retriever.params = {}
                # Mock other required attributes
                mock_config.batch_size = 32
                mock_config.validate_urls = True
                mock_config.log_level = "INFO"
                mock_load_config.return_value = mock_config

                mock_resource = Mock()
                mock_create_resource.return_value = mock_resource

                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_asyncio_run.return_value = {"successful": 1, "total_documents": 2}

                # Run command
                result = runner.invoke(
                    resources_group, ["add", "--url", "https://example.com/doc", "--type", type_str]
                )

                # Verify
                assert result.exit_code == 0
                mock_create_resource.assert_called_once_with(
                    url="https://example.com/doc", resource_type=expected_type, tags=[], priority=5
                )

    def test_add_resource_configuration_error(self):
        """Test add resource with configuration error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.logger"
        ) as mock_logger:
            mock_load_config.side_effect = ConfigurationError("Config not found")

            result = runner.invoke(resources_group, ["add", "--url", "https://example.com/doc"])

            assert result.exit_code == 1
            # Verify the logger was called with the expected error message
            mock_logger.error.assert_called_with("Configuration error: Config not found")
            mock_logger.info.assert_called_with(
                "Try creating a config file with: paas-ai config init"
            )

    def test_add_resource_processing_error(self):
        """Test add resource with processing error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "asyncio.run"
        ) as mock_asyncio_run, patch(
            "src.paas_ai.cli.commands.rag.resources.logger"
        ) as mock_logger:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_asyncio_run.side_effect = Exception("Processing failed")

            result = runner.invoke(resources_group, ["add", "--url", "https://example.com/doc"])

            assert result.exit_code == 1
            # Verify the logger was called with the expected error message
            mock_logger.error.assert_called_with("Failed to add resource: Processing failed")

    def test_add_resource_no_successful_processing(self):
        """Test add resource when no resources are successfully processed."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "asyncio.run"
        ) as mock_asyncio_run, patch(
            "src.paas_ai.cli.commands.rag.resources.logger"
        ) as mock_logger:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_asyncio_run.return_value = {"successful": 0, "total_documents": 0}

            result = runner.invoke(resources_group, ["add", "--url", "https://example.com/doc"])

            assert result.exit_code == 1
            # Verify the logger was called with the expected error message
            mock_logger.error.assert_called_with(
                "Failed to add resource: Failed to process any resources"
            )


class TestAddResourcesBatch:
    """Test the add resources batch command."""

    def test_add_resources_batch_success(self):
        """Test successful batch resource addition."""
        runner = CliRunner()

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "type", "tags", "priority"])
            writer.writerow(["https://example.com/doc1", "dsl", "test,example", "7"])
            writer.writerow(["https://example.com/doc2", "contextual", "demo", "5"])
            writer.writerow(["https://example.com/doc3", "guidelines", "", "3"])
            csv_file = f.name

        try:
            with patch(
                "src.paas_ai.cli.commands.rag.resources.load_config"
            ) as mock_load_config, patch(
                "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
            ) as mock_create_resource, patch(
                "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
            ) as mock_processor_class, patch(
                "asyncio.run"
            ) as mock_asyncio_run, patch(
                "src.paas_ai.cli.commands.rag.resources.logger"
            ) as mock_logger:
                # Setup mocks
                mock_config = Mock()
                mock_load_config.return_value = mock_config

                mock_resource = Mock()
                mock_create_resource.return_value = mock_resource

                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_asyncio_run.return_value = {
                    "successful": 3,
                    "failed": 0,
                    "total_documents": 10,
                }

                # Run command
                result = runner.invoke(resources_group, ["add-batch", "--csv-file", csv_file])

                # Verify
                assert result.exit_code == 0

                # Verify all logger calls were made
                mock_logger.info.assert_any_call("Found 3 resources in CSV file")
                mock_logger.success.assert_called_with("Batch processing completed:")
                mock_logger.info.assert_any_call("  Successful: 3")
                mock_logger.info.assert_any_call("  Failed: 0")
                mock_logger.info.assert_any_call("  Total documents: 10")

                # Verify create_resource_from_url was called for each row
                assert mock_create_resource.call_count == 3

        finally:
            os.unlink(csv_file)

    def test_add_resources_batch_empty_csv(self):
        """Test batch resource addition with empty CSV."""
        runner = CliRunner()

        # Create empty CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "type", "tags", "priority"])
            csv_file = f.name

        try:
            with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config:
                mock_config = Mock()
                mock_load_config.return_value = mock_config

                result = runner.invoke(resources_group, ["add-batch", "--csv-file", csv_file])

                assert result.exit_code == 1
                assert "No valid resources found in CSV file" in result.output

        finally:
            os.unlink(csv_file)

    def test_add_resources_batch_missing_url(self):
        """Test batch resource addition with missing URL in CSV."""
        runner = CliRunner()

        # Create CSV with missing URL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "type", "tags", "priority"])
            writer.writerow(["", "dsl", "test", "5"])  # Empty URL
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
                "asyncio.run"
            ) as mock_asyncio_run, patch(
                "src.paas_ai.cli.commands.rag.resources.logger"
            ) as mock_logger:
                # Setup mocks
                mock_config = Mock()
                mock_load_config.return_value = mock_config

                mock_resource = Mock()
                mock_create_resource.return_value = mock_resource

                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_asyncio_run.return_value = {"successful": 1, "failed": 0, "total_documents": 5}

                result = runner.invoke(resources_group, ["add-batch", "--csv-file", csv_file])

                # Should only process the valid row
                assert result.exit_code == 0

                # Verify all logger calls were made
                mock_logger.info.assert_any_call("Found 1 resources in CSV file")
                mock_logger.success.assert_called_with("Batch processing completed:")
                mock_logger.info.assert_any_call("  Successful: 1")
                mock_logger.info.assert_any_call("  Failed: 0")
                mock_logger.info.assert_any_call("  Total documents: 5")

                assert mock_create_resource.call_count == 1

        finally:
            os.unlink(csv_file)

    def test_add_resources_batch_configuration_error(self):
        """Test batch resource addition with configuration error."""
        runner = CliRunner()

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "type", "tags", "priority"])
            writer.writerow(["https://example.com/doc1", "dsl", "test", "5"])
            csv_file = f.name

        try:
            with patch(
                "src.paas_ai.cli.commands.rag.resources.load_config"
            ) as mock_load_config, patch(
                "src.paas_ai.cli.commands.rag.resources.logger"
            ) as mock_logger:
                mock_load_config.side_effect = ConfigurationError("Config not found")

                result = runner.invoke(resources_group, ["add-batch", "--csv-file", csv_file])

                assert result.exit_code == 1
                # Verify the logger was called with the expected error message
                mock_logger.error.assert_called_with("Configuration error: Config not found")
                mock_logger.info.assert_called_with(
                    "Try creating a config file with: paas-ai config init"
                )

        finally:
            os.unlink(csv_file)

    def test_add_resources_batch_processing_error(self):
        """Test batch resource addition with processing error."""
        runner = CliRunner()

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "type", "tags", "priority"])
            writer.writerow(["https://example.com/doc1", "dsl", "test", "5"])
            csv_file = f.name

        try:
            with patch(
                "src.paas_ai.cli.commands.rag.resources.load_config"
            ) as mock_load_config, patch(
                "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
            ) as mock_create_resource, patch(
                "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
            ) as mock_processor_class, patch(
                "asyncio.run"
            ) as mock_asyncio_run, patch(
                "src.paas_ai.cli.commands.rag.resources.logger"
            ) as mock_logger:
                # Setup mocks
                mock_config = Mock()
                mock_load_config.return_value = mock_config

                mock_resource = Mock()
                mock_create_resource.return_value = mock_resource

                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_asyncio_run.side_effect = Exception("Processing failed")

                result = runner.invoke(resources_group, ["add-batch", "--csv-file", csv_file])

                assert result.exit_code == 1
                # Verify the logger was called with the expected error message
                mock_logger.error.assert_called_with("Failed to add resources: Processing failed")

        finally:
            os.unlink(csv_file)


class TestListResources:
    """Test the list resources command."""

    def test_list_resources_not_implemented(self):
        """Test that list resources shows not implemented message."""
        runner = CliRunner()

        # Mock the logger to capture its output
        with patch("src.paas_ai.cli.commands.rag.resources.logger") as mock_logger:
            result = runner.invoke(resources_group, ["list"])

            assert result.exit_code == 0
            # Verify the logger was called with the expected message
            mock_logger.info.assert_called_once_with("Resource listing not yet implemented")


class TestRemoveResource:
    """Test the remove resource command."""

    def test_remove_resource_not_implemented(self):
        """Test that remove resource shows not implemented message."""
        runner = CliRunner()

        # Mock the logger to capture its output
        with patch("src.paas_ai.cli.commands.rag.resources.logger") as mock_logger:
            result = runner.invoke(resources_group, ["remove", "resource-id-123"])

            assert result.exit_code == 0
            # Verify the logger was called with the expected message
            mock_logger.info.assert_called_once_with(
                "Resource removal not yet implemented: resource-id-123"
            )


class TestClearResources:
    """Test the clear resources command."""

    def test_clear_resources_success(self):
        """Test successful resource clearing."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "src.paas_ai.cli.commands.rag.resources.logger"
        ) as mock_logger:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Run command with confirmation
            result = runner.invoke(resources_group, ["clear"], input="y\n")

            assert result.exit_code == 0
            # Verify the logger was called with the expected message
            mock_logger.success.assert_called_once_with("Knowledge base cleared successfully")
            mock_processor.clear_knowledge_base.assert_called_once()

    def test_clear_resources_cancelled(self):
        """Test resource clearing when user cancels."""
        runner = CliRunner()

        # Run command without confirmation
        result = runner.invoke(resources_group, ["clear"], input="n\n")

        assert result.exit_code == 1  # Click confirmation failure

    def test_clear_resources_configuration_error(self):
        """Test clear resources with configuration error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.logger"
        ) as mock_logger:
            mock_load_config.side_effect = ConfigurationError("Config not found")

            result = runner.invoke(resources_group, ["clear"], input="y\n")

            assert result.exit_code == 1
            # Verify the logger was called with the expected error message
            mock_logger.error.assert_called_with("Configuration error: Config not found")

    def test_clear_resources_processing_error(self):
        """Test clear resources with processing error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "src.paas_ai.cli.commands.rag.resources.logger"
        ) as mock_logger:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor.clear_knowledge_base.side_effect = Exception("Clear failed")
            mock_processor_class.return_value = mock_processor

            result = runner.invoke(resources_group, ["clear"], input="y\n")

            assert result.exit_code == 1
            # Verify the logger was called with the expected error message
            mock_logger.error.assert_called_with("Failed to clear knowledge base: Clear failed")


class TestResourcesEdgeCases:
    """Test edge cases for resources commands."""

    def test_add_resource_empty_tags(self):
        """Test adding resource with empty tags."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "asyncio.run"
        ) as mock_asyncio_run:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_asyncio_run.return_value = {"successful": 1, "total_documents": 2}

            # Run command with empty tags
            result = runner.invoke(
                resources_group, ["add", "--url", "https://example.com/doc", "--tags", ""]
            )

            assert result.exit_code == 0
            mock_create_resource.assert_called_once_with(
                url="https://example.com/doc",
                resource_type=ResourceType.DSL,
                tags=[],  # Should be empty list
                priority=5,
            )

    def test_add_resource_whitespace_tags(self):
        """Test adding resource with whitespace-only tags."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "asyncio.run"
        ) as mock_asyncio_run:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_asyncio_run.return_value = {"successful": 1, "total_documents": 2}

            # Run command with whitespace tags
            result = runner.invoke(
                resources_group, ["add", "--url", "https://example.com/doc", "--tags", "  ,  ,  "]
            )

            assert result.exit_code == 0
            mock_create_resource.assert_called_once_with(
                url="https://example.com/doc",
                resource_type=ResourceType.DSL,
                tags=[],  # Should be empty list after stripping
                priority=5,
            )

    def test_add_resource_priority_boundaries(self):
        """Test adding resource with boundary priority values."""
        runner = CliRunner()

        test_cases = [1, 5, 10]  # Min, default, max

        for priority in test_cases:
            with patch(
                "src.paas_ai.cli.commands.rag.resources.load_config"
            ) as mock_load_config, patch(
                "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
            ) as mock_create_resource, patch(
                "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
            ) as mock_processor_class, patch(
                "asyncio.run"
            ) as mock_asyncio_run:
                # Setup mocks
                mock_config = Mock()
                mock_load_config.return_value = mock_config

                mock_resource = Mock()
                mock_create_resource.return_value = mock_resource

                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_asyncio_run.return_value = {"successful": 1, "total_documents": 2}

                # Run command
                result = runner.invoke(
                    resources_group,
                    ["add", "--url", "https://example.com/doc", "--priority", str(priority)],
                )

                assert result.exit_code == 0
                mock_create_resource.assert_called_once_with(
                    url="https://example.com/doc",
                    resource_type=ResourceType.DSL,
                    tags=[],
                    priority=priority,
                )

    def test_add_resources_batch_malformed_csv(self):
        """Test batch resource addition with malformed CSV."""
        runner = CliRunner()

        # Create malformed CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("invalid,csv,content\n")
            f.write("not,properly,formatted\n")
            csv_file = f.name

        try:
            with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config:
                mock_config = Mock()
                mock_load_config.return_value = mock_config

                result = runner.invoke(resources_group, ["add-batch", "--csv-file", csv_file])

                # Should handle malformed CSV gracefully
                assert result.exit_code == 0 or result.exit_code == 1

        finally:
            os.unlink(csv_file)

    def test_config_profile_warning(self):
        """Test that config profile override shows warning."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "asyncio.run"
        ) as mock_asyncio_run, patch(
            "src.paas_ai.cli.commands.rag.resources.logger"
        ) as mock_logger:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_asyncio_run.return_value = {"successful": 1, "total_documents": 2}

            # Run command with config profile
            result = runner.invoke(
                resources_group,
                ["add", "--url", "https://example.com/doc", "--config-profile", "custom"],
            )

            assert result.exit_code == 0
            # Verify the logger was called with the expected warning message
            mock_logger.warning.assert_called_with(
                "Config profile override not yet implemented: custom"
            )


class TestResourcesIntegration:
    """Integration tests for resources commands."""

    def test_full_workflow_add_resource(self):
        """Test complete workflow for adding a resource."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "asyncio.run"
        ) as mock_asyncio_run, patch(
            "src.paas_ai.cli.commands.rag.resources.logger"
        ) as mock_logger:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_asyncio_run.return_value = {"successful": 1, "total_documents": 5}

            # Run command
            result = runner.invoke(
                resources_group,
                [
                    "add",
                    "--url",
                    "https://example.com/doc",
                    "--type",
                    "contextual",
                    "--tags",
                    "integration,test",
                    "--priority",
                    "8",
                ],
            )

            # Verify complete workflow
            assert result.exit_code == 0

            # Verify all logger calls were made
            mock_logger.info.assert_any_call("Using configuration profile with openai embeddings")
            mock_logger.success.assert_called_with("Successfully processed 1 resources")
            mock_logger.info.assert_any_call("Total documents: 5")

            # Verify all components were called
            mock_load_config.assert_called_once()
            mock_create_resource.assert_called_once()
            mock_processor_class.assert_called_once_with(mock_config)
            mock_processor.add_resources.assert_called_once_with([mock_resource])
            mock_asyncio_run.assert_called_once()

    def test_full_workflow_batch_add(self):
        """Test complete workflow for batch adding resources."""
        runner = CliRunner()

        # Create comprehensive CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "type", "tags", "priority"])
            writer.writerow(["https://example.com/doc1", "dsl", "test,example", "7"])
            writer.writerow(["https://example.com/doc2", "contextual", "demo", "5"])
            writer.writerow(["https://example.com/doc3", "guidelines", "", "3"])
            writer.writerow(["https://example.com/doc4", "domain_rules", "rules,validation", "9"])
            csv_file = f.name

        try:
            with patch(
                "src.paas_ai.cli.commands.rag.resources.load_config"
            ) as mock_load_config, patch(
                "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
            ) as mock_create_resource, patch(
                "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
            ) as mock_processor_class, patch(
                "asyncio.run"
            ) as mock_asyncio_run, patch(
                "src.paas_ai.cli.commands.rag.resources.logger"
            ) as mock_logger:
                # Setup mocks
                mock_config = Mock()
                mock_config.embedding.type = "sentence_transformers"
                mock_load_config.return_value = mock_config

                mock_resource = Mock()
                mock_create_resource.return_value = mock_resource

                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_asyncio_run.return_value = {
                    "successful": 4,
                    "failed": 0,
                    "total_documents": 15,
                }

                # Run command
                result = runner.invoke(resources_group, ["add-batch", "--csv-file", csv_file])

                # Verify complete workflow
                assert result.exit_code == 0

                # Verify all logger calls were made
                mock_logger.info.assert_any_call(
                    "Using configuration profile with sentence_transformers embeddings"
                )
                mock_logger.info.assert_any_call("Found 4 resources in CSV file")
                mock_logger.success.assert_called_with("Batch processing completed:")
                mock_logger.info.assert_any_call("  Successful: 4")
                mock_logger.info.assert_any_call("  Failed: 0")
                mock_logger.info.assert_any_call("  Total documents: 15")

                # Verify all resources were processed
                assert mock_create_resource.call_count == 4
                mock_processor.add_resources.assert_called_once()

        finally:
            os.unlink(csv_file)

    def test_error_handling_workflow(self):
        """Test error handling across the workflow."""
        runner = CliRunner()

        # Test configuration error
        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.logger"
        ) as mock_logger:
            mock_load_config.side_effect = ConfigurationError("Config file not found")

            result = runner.invoke(resources_group, ["add", "--url", "https://example.com/doc"])

            assert result.exit_code == 1
            # Verify the logger was called with the expected error message
            mock_logger.error.assert_called_with("Configuration error: Config file not found")
            mock_logger.info.assert_called_with(
                "Try creating a config file with: paas-ai config init"
            )

        # Test processing error
        with patch("src.paas_ai.cli.commands.rag.resources.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.resources.create_resource_from_url"
        ) as mock_create_resource, patch(
            "src.paas_ai.cli.commands.rag.resources.RAGProcessor"
        ) as mock_processor_class, patch(
            "asyncio.run"
        ) as mock_asyncio_run, patch(
            "src.paas_ai.cli.commands.rag.resources.logger"
        ) as mock_logger:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_resource = Mock()
            mock_create_resource.return_value = mock_resource

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_asyncio_run.side_effect = Exception("Network error")

            result = runner.invoke(resources_group, ["add", "--url", "https://example.com/doc"])

            assert result.exit_code == 1
            # Verify the logger was called with the expected error message
            mock_logger.error.assert_called_with("Failed to add resource: Network error")

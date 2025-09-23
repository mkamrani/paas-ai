"""
Unit tests for RAG sync CLI command.

Tests all components of the sync CLI module including:
- Sync command functionality
- Resource type filtering
- Incremental vs full sync modes
- Force sync option
- Resource ID specific sync
- Dry run mode
- Error handling and edge cases
- Progress reporting
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from src.paas_ai.cli.commands.rag.sync import sync


class TestSyncCommand:
    """Test the sync command."""

    def test_sync_basic_incremental(self):
        """Test basic incremental sync."""
        runner = CliRunner()

        # Mock the logger to capture behavior instead of relying on output
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, ["--incremental"])

            # Verify command succeeded
            assert result.exit_code == 0

            # Verify logger behavior instead of output
            mock_logger.set_context.assert_called_once_with("RAG-SYNC")
            mock_logger.info.assert_any_call("Starting incremental synchronization")
            mock_logger.info.assert_any_call("Syncing all resources")
            mock_logger.progress.assert_any_call("Analyzing resources for changes...")
            mock_logger.progress.assert_any_call("Loading configurations...")
            mock_logger.info.assert_any_call("Found 5 resources to sync")
            mock_logger.progress.assert_any_call("Fetching updated content...")
            mock_logger.progress.assert_any_call("Processing documents...")
            mock_logger.progress.assert_any_call("Updating vector indices...")
            mock_logger.progress.assert_any_call("Updating metadata...")
            mock_logger.success.assert_called_once_with(
                "Synchronization completed - 5 resources updated"
            )
            mock_logger.clear_context.assert_called_once()

    def test_sync_full_mode(self):
        """Test full sync mode (default)."""
        runner = CliRunner()

        # Mock the logger to capture behavior instead of relying on output
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, [])

            # Verify command succeeded
            assert result.exit_code == 0

            # Verify logger behavior instead of output
            mock_logger.info.assert_any_call("Starting full synchronization")
            mock_logger.info.assert_any_call("Syncing all resources")
            mock_logger.info.assert_any_call("Found 5 resources to sync")
            mock_logger.success.assert_called_once_with(
                "Synchronization completed - 5 resources updated"
            )

    def test_sync_forced_mode(self):
        """Test forced sync mode."""
        runner = CliRunner()

        # Mock the logger to capture behavior instead of relying on output
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, ["--force"])

            # Verify command succeeded
            assert result.exit_code == 0

            # Verify logger behavior instead of output
            mock_logger.info.assert_any_call("Starting forced synchronization")
            mock_logger.info.assert_any_call("Syncing all resources")
            mock_logger.info.assert_any_call("Found 5 resources to sync")
            mock_logger.success.assert_called_once_with(
                "Synchronization completed - 5 resources updated"
            )

    def test_sync_with_resource_type_filter(self):
        """Test sync with specific resource type filter."""
        runner = CliRunner()

        test_cases = [
            ("dsl", "dsl"),
            ("contextual", "contextual"),
            ("guidelines", "guidelines"),
            ("domain_rules", "domain_rules"),
        ]

        for type_str, expected_type in test_cases:
            result = runner.invoke(sync, ["--type", type_str])

            # Verify
            assert result.exit_code == 0
            assert f"Starting full synchronization" in result.output
            assert f"Syncing {expected_type} resources" in result.output
            assert "Found 5 resources to sync" in result.output
            assert "Synchronization completed - 5 resources updated" in result.output

    def test_sync_with_resource_id(self):
        """Test sync with specific resource ID."""
        runner = CliRunner()

        result = runner.invoke(sync, ["--resource-id", "resource-123"])

        # Verify
        assert result.exit_code == 0
        assert "Starting full synchronization" in result.output
        assert "Syncing specific resource: resource-123" in result.output
        assert "Found 5 resources to sync" in result.output
        assert "Synchronization completed - 5 resources updated" in result.output

    def test_sync_dry_run_mode(self):
        """Test sync in dry run mode."""
        runner = CliRunner()

        result = runner.invoke(sync, ["--dry-run"])

        # Verify
        assert result.exit_code == 0
        assert "DRY RUN MODE - No actual sync will occur" in result.output
        assert "Starting full synchronization" in result.output
        assert "Syncing all resources" in result.output
        assert "Found 5 resources to sync" in result.output
        assert "Dry run completed - 5 resources would be updated" in result.output

    def test_sync_incremental_no_changes(self):
        """Test incremental sync when no changes are detected."""
        runner = CliRunner()

        # This test would need to mock the changes_detected logic
        # For now, we test the current behavior where it always shows 5 changes
        result = runner.invoke(sync, ["--incremental"])

        # Verify
        assert result.exit_code == 0
        assert "Starting incremental synchronization" in result.output
        assert "Found 5 resources to sync" in result.output
        assert "Synchronization completed - 5 resources updated" in result.output

    def test_sync_combined_options(self):
        """Test sync with multiple options combined."""
        runner = CliRunner()

        result = runner.invoke(sync, ["--type", "dsl", "--incremental", "--dry-run"])

        # Verify
        assert result.exit_code == 0
        assert "DRY RUN MODE - No actual sync will occur" in result.output
        assert "Starting incremental synchronization" in result.output
        assert "Syncing dsl resources" in result.output
        assert "Found 5 resources to sync" in result.output
        assert "Dry run completed - 5 resources would be updated" in result.output

    def test_sync_force_with_incremental(self):
        """Test sync with both force and incremental flags."""
        runner = CliRunner()

        result = runner.invoke(sync, ["--incremental", "--force"])

        # Verify
        assert result.exit_code == 0
        assert "Starting forced synchronization" in result.output  # Force takes precedence
        assert "Syncing all resources" in result.output
        assert "Found 5 resources to sync" in result.output
        assert "Synchronization completed - 5 resources updated" in result.output

    def test_sync_resource_id_with_type(self):
        """Test sync with both resource ID and type specified."""
        runner = CliRunner()

        result = runner.invoke(sync, ["--resource-id", "resource-456", "--type", "contextual"])

        # Verify
        assert result.exit_code == 0
        assert "Starting full synchronization" in result.output
        assert "Syncing specific resource: resource-456" in result.output
        # Note: The current implementation prioritizes resource-id over type
        assert "Found 5 resources to sync" in result.output
        assert "Synchronization completed - 5 resources updated" in result.output


class TestSyncEdgeCases:
    """Test edge cases for sync command."""

    def test_sync_with_empty_resource_id(self):
        """Test sync with empty resource ID."""
        runner = CliRunner()

        # Mock the logger to capture behavior instead of relying on output
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, ["--resource-id", ""])

            # Verify command succeeded
            assert result.exit_code == 0

            # Verify logger behavior instead of output
            # Empty string is falsy, so it goes to the else branch
            mock_logger.info.assert_any_call("Starting full synchronization")
            mock_logger.info.assert_any_call("Syncing all resources")  # Empty string is falsy
            mock_logger.info.assert_any_call("Found 5 resources to sync")
            mock_logger.success.assert_called_once_with(
                "Synchronization completed - 5 resources updated"
            )

    def test_sync_with_very_long_resource_id(self):
        """Test sync with very long resource ID."""
        runner = CliRunner()

        long_resource_id = "a" * 1000
        result = runner.invoke(sync, ["--resource-id", long_resource_id])

        # Verify
        assert result.exit_code == 0
        assert "Starting full synchronization" in result.output
        assert f"Syncing specific resource: {long_resource_id}" in result.output
        assert "Found 5 resources to sync" in result.output
        assert "Synchronization completed - 5 resources updated" in result.output

    def test_sync_with_special_characters_resource_id(self):
        """Test sync with special characters in resource ID."""
        runner = CliRunner()

        special_resource_id = "resource@#$%^&*()_+-=[]{}|;':\",./<>?`~"
        result = runner.invoke(sync, ["--resource-id", special_resource_id])

        # Verify
        assert result.exit_code == 0
        assert "Starting full synchronization" in result.output
        assert f"Syncing specific resource: {special_resource_id}" in result.output
        assert "Found 5 resources to sync" in result.output
        assert "Synchronization completed - 5 resources updated" in result.output

    def test_sync_with_unicode_resource_id(self):
        """Test sync with unicode characters in resource ID."""
        runner = CliRunner()

        unicode_resource_id = "资源-123-测试-🚀"
        result = runner.invoke(sync, ["--resource-id", unicode_resource_id])

        # Verify
        assert result.exit_code == 0
        assert "Starting full synchronization" in result.output
        assert f"Syncing specific resource: {unicode_resource_id}" in result.output
        assert "Found 5 resources to sync" in result.output
        assert "Synchronization completed - 5 resources updated" in result.output

    def test_sync_all_options_combined(self):
        """Test sync with all options combined."""
        runner = CliRunner()

        result = runner.invoke(
            sync,
            [
                "--type",
                "guidelines",
                "--incremental",
                "--force",
                "--resource-id",
                "test-resource",
                "--dry-run",
            ],
        )

        # Verify
        assert result.exit_code == 0
        assert "DRY RUN MODE - No actual sync will occur" in result.output
        assert "Starting forced synchronization" in result.output  # Force takes precedence
        assert (
            "Syncing specific resource: test-resource" in result.output
        )  # Resource ID takes precedence
        assert "Found 5 resources to sync" in result.output
        assert "Dry run completed - 5 resources would be updated" in result.output

    def test_sync_with_none_resource_id(self):
        """Test sync with None resource ID (should not happen in CLI but test edge case)."""
        runner = CliRunner()

        # This tests the case where resource_id might be None internally
        # The CLI should handle this gracefully
        result = runner.invoke(sync, [])

        # Verify
        assert result.exit_code == 0
        assert "Starting full synchronization" in result.output
        assert "Syncing all resources" in result.output
        assert "Found 5 resources to sync" in result.output
        assert "Synchronization completed - 5 resources updated" in result.output


class TestSyncErrorHandling:
    """Test error handling for sync command."""

    def test_sync_with_exception(self):
        """Test sync when an exception occurs."""
        runner = CliRunner()

        # Mock the logger to raise an exception
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_logger.set_context.side_effect = Exception("Logger error")
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, [])

            # Verify command failed with proper exit code
            assert result.exit_code == 1
            # When Click catches exceptions, the output may be empty or contain the exception
            # The important thing is that the command failed (exit code 1)

    def test_sync_with_logger_exception(self):
        """Test sync when logger operations fail."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_logger.info.side_effect = Exception("Info logging failed")
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, [])

            # Verify
            assert result.exit_code == 1
            assert "Sync failed" in result.output

    def test_sync_with_progress_exception(self):
        """Test sync when progress logging fails."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_logger.progress.side_effect = Exception("Progress logging failed")
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, [])

            # Verify
            assert result.exit_code == 1
            assert "Sync failed" in result.output

    def test_sync_with_success_exception(self):
        """Test sync when success logging fails."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_logger.success.side_effect = Exception("Success logging failed")
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, [])

            # Verify
            assert result.exit_code == 1
            assert "Sync failed" in result.output

    def test_sync_with_clear_context_exception(self):
        """Test sync when clear_context fails."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_logger.clear_context.side_effect = Exception("Clear context failed")
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, [])

            # Verify command failed with proper exit code
            assert result.exit_code == 1
            # When Click catches exceptions, the output may be empty or contain the exception
            # The important thing is that the command failed (exit code 1)


class TestSyncIntegration:
    """Integration tests for sync command."""

    def test_full_workflow_incremental_sync(self):
        """Test complete workflow for incremental sync."""
        runner = CliRunner()

        # Mock the logger to capture behavior instead of relying on output
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, ["--incremental"])

            # Verify complete workflow succeeded
            assert result.exit_code == 0

            # Verify logger behavior instead of output
            mock_logger.info.assert_any_call("Starting incremental synchronization")
            mock_logger.info.assert_any_call("Syncing all resources")
            mock_logger.progress.assert_any_call("Analyzing resources for changes...")
            mock_logger.progress.assert_any_call("Loading configurations...")
            mock_logger.info.assert_any_call("Found 5 resources to sync")
            mock_logger.progress.assert_any_call("Fetching updated content...")
            mock_logger.progress.assert_any_call("Processing documents...")
            mock_logger.progress.assert_any_call("Updating vector indices...")
            mock_logger.progress.assert_any_call("Updating metadata...")
            mock_logger.success.assert_called_once_with(
                "Synchronization completed - 5 resources updated"
            )

    def test_full_workflow_full_sync(self):
        """Test complete workflow for full sync."""
        runner = CliRunner()

        # Mock the logger to capture behavior instead of relying on output
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, [])

            # Verify complete workflow succeeded
            assert result.exit_code == 0

            # Verify logger behavior instead of output
            mock_logger.info.assert_any_call("Starting full synchronization")
            mock_logger.info.assert_any_call("Syncing all resources")
            mock_logger.progress.assert_any_call("Analyzing resources for changes...")
            mock_logger.progress.assert_any_call("Loading configurations...")
            mock_logger.info.assert_any_call("Found 5 resources to sync")
            mock_logger.progress.assert_any_call("Fetching updated content...")
            mock_logger.progress.assert_any_call("Processing documents...")
            mock_logger.progress.assert_any_call("Updating vector indices...")
            mock_logger.progress.assert_any_call("Updating metadata...")
            mock_logger.success.assert_called_once_with(
                "Synchronization completed - 5 resources updated"
            )

    def test_full_workflow_forced_sync(self):
        """Test complete workflow for forced sync."""
        runner = CliRunner()

        # Mock the logger to capture behavior instead of relying on output
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, ["--force"])

            # Verify complete workflow succeeded
            assert result.exit_code == 0

            # Verify logger behavior instead of output
            mock_logger.info.assert_any_call("Starting forced synchronization")
            mock_logger.info.assert_any_call("Syncing all resources")
            mock_logger.progress.assert_any_call("Analyzing resources for changes...")
            mock_logger.progress.assert_any_call("Loading configurations...")
            mock_logger.info.assert_any_call("Found 5 resources to sync")
            mock_logger.progress.assert_any_call("Fetching updated content...")
            mock_logger.progress.assert_any_call("Processing documents...")
            mock_logger.progress.assert_any_call("Updating vector indices...")
            mock_logger.progress.assert_any_call("Updating metadata...")
            mock_logger.success.assert_called_once_with(
                "Synchronization completed - 5 resources updated"
            )

    def test_full_workflow_dry_run(self):
        """Test complete workflow for dry run sync."""
        runner = CliRunner()

        # Mock the logger to capture behavior instead of relying on output
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, ["--dry-run"])

            # Verify complete workflow succeeded
            assert result.exit_code == 0

            # Verify logger behavior instead of output
            mock_logger.warning.assert_called_once_with("DRY RUN MODE - No actual sync will occur")
            mock_logger.info.assert_any_call("Starting full synchronization")
            mock_logger.info.assert_any_call("Syncing all resources")
            mock_logger.progress.assert_any_call("Analyzing resources for changes...")
            mock_logger.progress.assert_any_call("Loading configurations...")
            mock_logger.info.assert_any_call("Found 5 resources to sync")
            mock_logger.progress.assert_any_call("Fetching updated content...")
            mock_logger.progress.assert_any_call("Processing documents...")
            mock_logger.progress.assert_any_call("Updating vector indices...")
            mock_logger.progress.assert_any_call("Updating metadata...")
            mock_logger.info.assert_called_with("Dry run completed - 5 resources would be updated")

    def test_full_workflow_resource_type_sync(self):
        """Test complete workflow for resource type specific sync."""
        runner = CliRunner()

        # Mock the logger to capture behavior instead of relying on output
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, ["--type", "dsl"])

            # Verify complete workflow succeeded
            assert result.exit_code == 0

            # Verify logger behavior instead of output
            mock_logger.info.assert_any_call("Starting full synchronization")
            mock_logger.info.assert_any_call("Syncing dsl resources")
            mock_logger.progress.assert_any_call("Analyzing resources for changes...")
            mock_logger.progress.assert_any_call("Loading configurations...")
            mock_logger.info.assert_any_call("Found 5 resources to sync")
            mock_logger.progress.assert_any_call("Fetching updated content...")
            mock_logger.progress.assert_any_call("Processing documents...")
            mock_logger.progress.assert_any_call("Updating vector indices...")
            mock_logger.progress.assert_any_call("Updating metadata...")
            mock_logger.success.assert_called_once_with(
                "Synchronization completed - 5 resources updated"
            )

    def test_full_workflow_resource_id_sync(self):
        """Test complete workflow for resource ID specific sync."""
        runner = CliRunner()

        # Mock the logger to capture behavior instead of relying on output
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, ["--resource-id", "specific-resource-123"])

            # Verify complete workflow succeeded
            assert result.exit_code == 0

            # Verify logger behavior instead of output
            mock_logger.info.assert_any_call("Starting full synchronization")
            mock_logger.info.assert_any_call("Syncing specific resource: specific-resource-123")
            mock_logger.progress.assert_any_call("Analyzing resources for changes...")
            mock_logger.progress.assert_any_call("Loading configurations...")
            mock_logger.info.assert_any_call("Found 5 resources to sync")
            mock_logger.progress.assert_any_call("Fetching updated content...")
            mock_logger.progress.assert_any_call("Processing documents...")
            mock_logger.progress.assert_any_call("Updating vector indices...")
            mock_logger.progress.assert_any_call("Updating metadata...")
            mock_logger.success.assert_called_once_with(
                "Synchronization completed - 5 resources updated"
            )

    def test_error_handling_workflow(self):
        """Test error handling across the workflow."""
        runner = CliRunner()

        # Test with logger exception
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_logger.set_context.side_effect = Exception("Logger setup failed")
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, [])

            # Verify command failed with proper exit code
            assert result.exit_code == 1
            # When Click catches exceptions, the output may be empty or contain the exception
            # The important thing is that the command failed (exit code 1)

        # Test with progress logging exception
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_logger.progress.side_effect = Exception("Progress failed")
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, [])

            assert result.exit_code == 1
            assert "Sync failed" in result.output

    def test_sync_mode_priority(self):
        """Test that sync mode priority is correct."""
        runner = CliRunner()

        # Test force takes precedence over incremental
        result = runner.invoke(sync, ["--incremental", "--force"])
        assert result.exit_code == 0
        assert "Starting forced synchronization" in result.output

        # Test resource-id takes precedence over type
        result = runner.invoke(sync, ["--type", "dsl", "--resource-id", "test-resource"])
        assert result.exit_code == 0
        assert "Syncing specific resource: test-resource" in result.output

    def test_sync_progress_sequence(self):
        """Test that sync progress messages are in correct sequence."""
        runner = CliRunner()

        # Mock the logger to capture behavior instead of relying on output
        with patch("src.paas_ai.cli.commands.rag.sync.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = runner.invoke(sync, [])

            # Verify command succeeded
            assert result.exit_code == 0

            # Verify progress sequence through mock calls
            expected_progress_calls = [
                "Analyzing resources for changes...",
                "Loading configurations...",
                "Fetching updated content...",
                "Processing documents...",
                "Updating vector indices...",
                "Updating metadata...",
            ]

            # Verify all progress calls were made
            for expected_call in expected_progress_calls:
                mock_logger.progress.assert_any_call(expected_call)

    def test_sync_completion_messages(self):
        """Test that sync completion messages are correct."""
        runner = CliRunner()

        # Test normal completion
        result = runner.invoke(sync, [])
        assert result.exit_code == 0
        assert "Synchronization completed - 5 resources updated" in result.output

        # Test dry run completion
        result = runner.invoke(sync, ["--dry-run"])
        assert result.exit_code == 0
        assert "Dry run completed - 5 resources would be updated" in result.output

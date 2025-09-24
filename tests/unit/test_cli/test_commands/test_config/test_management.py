"""
Unit tests for CLI config profile management commands.

Tests all components of profile management commands including:
- set-current command
- add-profile command
- remove-profile command
- Profile validation
- Error handling and edge cases
- File system interactions
- User confirmations
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, mock_open, patch

import pytest
from click.testing import CliRunner

from src.paas_ai.cli.commands.config.main import add_profile, remove_profile, set_current_profile
from src.paas_ai.core.config import (
    Config,
    ConfigFile,
    ConfigFileNotFoundError,
    ConfigurationError,
    EmbeddingConfig,
    EmbeddingType,
    RetrieverConfig,
    RetrieverType,
    VectorStoreConfig,
    VectorStoreType,
)


class TestSetCurrentProfileCommand:
    """Test the set_current_profile command."""

    def test_set_current_profile_basic(self):
        """Test basic set current profile command."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
                "production": "Production environment profile",
            }

            mock_config_file = Mock()
            mock_config_file.current = "default"
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            result = runner.invoke(set_current_profile, ["production"])

            # Verify command succeeded
            assert result.exit_code == 0

            # Verify the profile was updated
            assert mock_config_file.current == "production"
            mock_save_file.assert_called_once_with(mock_config_file, mock_config_path)

    def test_set_current_profile_with_nonexistent_profile(self):
        """Test set current profile command with nonexistent profile."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.list_available_profiles") as mock_list:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
            }

            result = runner.invoke(set_current_profile, ["nonexistent"])

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0

    def test_set_current_profile_with_config_file_creation(self):
        """Test set current profile command with config file creation."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
            }

            # Simulate config file not found
            mock_load_file.side_effect = ConfigFileNotFoundError("Config file not found")
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")

            result = runner.invoke(set_current_profile, ["local"])

            # Verify command succeeded
            assert result.exit_code == 0

    def test_set_current_profile_with_same_profile(self):
        """Test set current profile command with same profile."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
            }

            mock_config_file = Mock()
            mock_config_file.current = "local"
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            result = runner.invoke(set_current_profile, ["local"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the profile was updated (even if same value)
            assert mock_config_file.current == "local"
            mock_save_file.assert_called_once_with(mock_config_file, mock_config_path)

    def test_set_current_profile_with_error(self):
        """Test set current profile command with error."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file:
            mock_list.return_value = {"default": "Default configuration profile"}

            mock_load_file.side_effect = Exception("Config load error")

            result = runner.invoke(set_current_profile, ["default"])

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0


class TestAddProfileCommand:
    """Test the add_profile command."""

    def test_add_profile_basic(self):
        """Test basic add profile command."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {}
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            result = runner.invoke(
                add_profile,
                [
                    "my-custom",
                    "--embedding-type",
                    "sentence_transformers",
                    "--embedding-model",
                    "all-MiniLM-L6-v2",
                    "--vectorstore-type",
                    "chroma",
                    "--vectorstore-dir",
                    "./rag_data/my-custom",
                    "--collection-name",
                    "my-custom_knowledge",
                    "--retriever-type",
                    "similarity",
                    "--batch-size",
                    "64",
                ],
            )

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify profile was added to config
            assert "my-custom" in mock_config_file.profiles
            mock_save_file.assert_called_once_with(mock_config_file, mock_config_path)

    def test_add_profile_with_activate(self):
        """Test add profile command with activate flag."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {}
            mock_config_file.current = "default"
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            result = runner.invoke(add_profile, ["my-custom", "--activate"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify profile was added and activated
            assert "my-custom" in mock_config_file.profiles
            assert mock_config_file.current == "my-custom"
            mock_save_file.assert_called_once_with(mock_config_file, mock_config_path)

    def test_add_profile_with_builtin_name_conflict(self):
        """Test add profile command with built-in name conflict."""
        runner = CliRunner()

        with patch("src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES") as mock_defaults:
            mock_defaults.__contains__ = lambda x, y: y == "default"
            mock_defaults.keys.return_value = ["default", "local", "production"]

            result = runner.invoke(add_profile, ["default"])

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0

    def test_add_profile_with_config_file_creation(self):
        """Test add profile command with config file creation."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            # Simulate config file not found
            mock_load_file.side_effect = ConfigFileNotFoundError("Config file not found")
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")

            result = runner.invoke(add_profile, ["my-custom"])

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0

    def test_add_profile_with_defaults(self):
        """Test add profile command with default values."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {}
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            result = runner.invoke(add_profile, ["my-custom"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify profile was added to config
            assert "my-custom" in mock_config_file.profiles
            mock_save_file.assert_called_once_with(mock_config_file, mock_config_path)

    def test_add_profile_with_error(self):
        """Test add profile command with error."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file:
            mock_defaults.__contains__ = lambda x, y: False
            mock_load_file.side_effect = Exception("Config load error")

            result = runner.invoke(add_profile, ["my-custom"])

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0


class TestRemoveProfileCommand:
    """Test the remove_profile command."""

    def test_remove_profile_basic(self):
        """Test basic remove profile command."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {"my-custom": Mock()}
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            result = runner.invoke(remove_profile, ["my-custom", "--confirm"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify profile was removed from config
            assert "my-custom" not in mock_config_file.profiles
            mock_save_file.assert_called_once_with(mock_config_file, mock_config_path)

    def test_remove_profile_with_confirmation(self):
        """Test remove profile command with confirmation prompt."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {"my-custom": Mock()}
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            # Simulate user confirming
            result = runner.invoke(remove_profile, ["my-custom"], input="y\n")

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify confirmation prompt was shown
            assert "Are you sure you want to remove profile 'my-custom'?" in result.output
            # Verify profile was removed from config
            assert "my-custom" not in mock_config_file.profiles
            mock_save_file.assert_called_once_with(mock_config_file, mock_config_path)

    def test_remove_profile_with_confirmation_cancelled(self):
        """Test remove profile command with confirmation cancelled."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {"my-custom": Mock()}
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            # Simulate user cancelling
            result = runner.invoke(remove_profile, ["my-custom"], input="n\n")

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify confirmation prompt was shown
            assert "Are you sure you want to remove profile 'my-custom'?" in result.output
            # Verify profile was NOT removed from config (cancelled)
            assert "my-custom" in mock_config_file.profiles

    def test_remove_profile_with_builtin_profile(self):
        """Test remove profile command with built-in profile."""
        runner = CliRunner()

        with patch("src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES") as mock_defaults:
            mock_defaults.__contains__ = lambda x, y: y == "default"

            result = runner.invoke(remove_profile, ["default"])

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0

    def test_remove_profile_with_nonexistent_profile(self):
        """Test remove profile command with nonexistent profile."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {}
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            result = runner.invoke(remove_profile, ["nonexistent"])

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0

    def test_remove_profile_with_current_profile(self):
        """Test remove profile command with current profile."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {"my-custom": Mock()}
            mock_config_file.current = "my-custom"
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            result = runner.invoke(remove_profile, ["my-custom", "--confirm"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify profile was removed and current profile was switched
            assert "my-custom" not in mock_config_file.profiles
            assert mock_config_file.current == "default"
            mock_save_file.assert_called_once_with(mock_config_file, mock_config_path)

    def test_remove_profile_with_config_file_not_found(self):
        """Test remove profile command with config file not found."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file:
            mock_defaults.__contains__ = lambda x, y: False
            mock_load_file.side_effect = ConfigFileNotFoundError("Config file not found")

            result = runner.invoke(remove_profile, ["my-custom"])

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0

    def test_remove_profile_with_error(self):
        """Test remove profile command with error."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file:
            mock_defaults.__contains__ = lambda x, y: False
            mock_load_file.side_effect = Exception("Config load error")

            result = runner.invoke(remove_profile, ["my-custom"])

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0


class TestProfileManagementCommandEdgeCases:
    """Test edge cases for profile management commands."""

    def test_set_current_profile_with_special_characters(self):
        """Test set current profile command with special characters."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_list.return_value = {
                "test-with-dashes": "Custom profile: Test with dashes",
                "test_with_underscores": "Custom profile: Test with underscores",
            }

            mock_config_file = Mock()
            mock_config_file.current = "default"
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            result = runner.invoke(set_current_profile, ["test-with-dashes"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the profile was updated
            assert mock_config_file.current == "test-with-dashes"
            mock_save_file.assert_called_once_with(mock_config_file, mock_config_path)

    def test_add_profile_with_unicode_characters(self):
        """Test add profile command with unicode characters."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {}
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            result = runner.invoke(add_profile, ["测试"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify profile was added to config
            assert "测试" in mock_config_file.profiles
            mock_save_file.assert_called_once_with(mock_config_file, mock_config_path)

    def test_remove_profile_with_very_long_name(self):
        """Test remove profile command with very long name."""
        runner = CliRunner()

        long_name = "a" * 100
        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {long_name: Mock()}
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            result = runner.invoke(remove_profile, [long_name, "--confirm"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify profile was removed from config
            assert long_name not in mock_config_file.profiles
            mock_save_file.assert_called_once_with(mock_config_file, mock_config_path)


class TestProfileManagementCommandIntegration:
    """Integration tests for profile management commands."""

    def test_profile_management_full_workflow(self):
        """Test complete profile management workflow."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            # Mock initial state
            mock_config_file = Mock()
            mock_config_file.profiles = {}
            mock_config_file.current = "default"
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            # Step 1: Add a profile
            result1 = runner.invoke(add_profile, ["workflow-test", "--activate"])
            assert result1.exit_code == 0
            # Verify profile was added and activated
            assert "workflow-test" in mock_config_file.profiles
            assert mock_config_file.current == "workflow-test"

            # Step 2: Set current profile back to default
            mock_list.return_value = {
                "default": "Default configuration profile",
                "workflow-test": "Custom profile: Workflow test profile",
            }

            result2 = runner.invoke(set_current_profile, ["default"])
            assert result2.exit_code == 0
            # Verify profile was switched
            assert mock_config_file.current == "default"

            # Step 3: Remove the profile
            result3 = runner.invoke(remove_profile, ["workflow-test", "--confirm"])
            assert result3.exit_code == 0
            # Verify profile was removed
            assert "workflow-test" not in mock_config_file.profiles

    def test_profile_management_consistency_across_calls(self):
        """Test that profile management commands are consistent across multiple calls."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {}
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            # Make multiple add profile calls
            result1 = runner.invoke(add_profile, ["test1"])
            result2 = runner.invoke(add_profile, ["test2"])

            assert result1.exit_code == 0
            assert result2.exit_code == 0
            assert "test1" in mock_config_file.profiles
            assert "test2" in mock_config_file.profiles


class TestProfileManagementCommandPerformance:
    """Test performance characteristics of profile management commands."""

    def test_set_current_profile_response_time(self):
        """Test that set current profile command responds quickly."""
        import time

        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
            }

            mock_config_file = Mock()
            mock_config_file.current = "default"
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            start_time = time.time()
            result = runner.invoke(set_current_profile, ["local"])
            end_time = time.time()

            # Should respond quickly (less than 1 second)
            assert (end_time - start_time) < 1.0
            assert result.exit_code == 0

    def test_add_profile_response_time(self):
        """Test that add profile command responds quickly."""
        import time

        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {}
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            start_time = time.time()
            result = runner.invoke(add_profile, ["performance-test"])
            end_time = time.time()

            # Should respond quickly (less than 1 second)
            assert (end_time - start_time) < 1.0
            assert result.exit_code == 0

    def test_remove_profile_response_time(self):
        """Test that remove profile command responds quickly."""
        import time

        runner = CliRunner()

        with patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file:
            mock_defaults.__contains__ = lambda x, y: False

            mock_config_file = Mock()
            mock_config_file.profiles = {"performance-test": Mock()}
            mock_config_path = Path("/home/user/.paas-ai/config.yaml")
            mock_load_file.return_value = (mock_config_file, mock_config_path)

            start_time = time.time()
            result = runner.invoke(remove_profile, ["performance-test", "--confirm"])
            end_time = time.time()

            # Should respond quickly (less than 1 second)
            assert (end_time - start_time) < 1.0
            assert result.exit_code == 0

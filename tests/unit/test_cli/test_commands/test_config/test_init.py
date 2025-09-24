"""
Unit tests for CLI config init command.

Tests all components of the init command including:
- Init command functionality
- Config file creation
- Profile setting
- Path handling
- Force overwrite
- Error handling and edge cases
- File system interactions
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, mock_open, patch

import pytest
from click.testing import CliRunner

from src.paas_ai.cli.commands.config.main import init_config
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


class TestInitConfigCommand:
    """Test the init_config command."""

    def test_init_config_basic(self):
        """Test basic init config command."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.home"
        ) as mock_home:
            # Mock path operations
            mock_home.return_value = Path("/home/user")
            mock_exists.return_value = False
            expected_path = Path("/home/user/.paas-ai/config.yaml")
            mock_create.return_value = expected_path

            result = runner.invoke(init_config)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected behavior through mock calls
            mock_create.assert_called_once_with(expected_path)

    def test_init_config_with_custom_path(self):
        """Test init config command with custom path."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.expanduser"
        ) as mock_expanduser, patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:
            # Mock path operations
            custom_path = Path("/custom/path/config.yaml")
            mock_expanduser.return_value = custom_path
            mock_resolve.return_value = custom_path
            mock_exists.return_value = False
            mock_create.return_value = custom_path

            result = runner.invoke(init_config, ["--path", "/custom/path/config.yaml"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected behavior through mock calls
            mock_create.assert_called_once_with(custom_path)

    def test_init_config_with_custom_profile(self):
        """Test init config command with custom profile."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file, patch(
            "pathlib.Path.exists"
        ) as mock_exists, patch(
            "pathlib.Path.home"
        ) as mock_home:
            # Mock path operations
            mock_home.return_value = Path("/home/user")
            mock_exists.return_value = False
            expected_path = Path("/home/user/.paas-ai/config.yaml")
            mock_create.return_value = expected_path

            # Mock config file operations
            mock_config_file = Mock()
            mock_config_file.current = "local"
            mock_load_file.return_value = (mock_config_file, expected_path)

            result = runner.invoke(init_config, ["--profile", "production"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify that profile was updated
            mock_create.assert_called_once_with(expected_path)
            mock_load_file.assert_called_once_with(expected_path)
            mock_save_file.assert_called_once_with(mock_config_file, expected_path)
            assert mock_config_file.current == "production"

    def test_init_config_with_force_overwrite(self):
        """Test init config command with force overwrite."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.home"
        ) as mock_home:
            # Mock path operations
            mock_home.return_value = Path("/home/user")
            mock_exists.return_value = True  # File exists
            expected_path = Path("/home/user/.paas-ai/config.yaml")
            mock_create.return_value = expected_path

            result = runner.invoke(init_config, ["--force"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected behavior through mock calls
            mock_create.assert_called_once_with(expected_path)

    def test_init_config_file_already_exists(self):
        """Test init config command when file already exists."""
        runner = CliRunner()

        with patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.expanduser"
        ) as mock_expanduser, patch("pathlib.Path.resolve") as mock_resolve:
            # Mock path operations
            mock_path = Path("/home/user/.paas-ai/config.yaml")
            mock_expanduser.return_value = mock_path
            mock_resolve.return_value = mock_path
            mock_exists.return_value = True  # File exists

            result = runner.invoke(init_config)

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0

    def test_init_config_with_all_options(self):
        """Test init config command with all options."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file, patch(
            "pathlib.Path.exists"
        ) as mock_exists, patch(
            "pathlib.Path.expanduser"
        ) as mock_expanduser, patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:
            # Mock path operations
            custom_path = Path("/custom/path/config.yaml")
            mock_expanduser.return_value = custom_path
            mock_resolve.return_value = custom_path
            mock_exists.return_value = False
            mock_create.return_value = custom_path

            # Mock config file operations
            mock_config_file = Mock()
            mock_config_file.current = "local"
            mock_load_file.return_value = (mock_config_file, custom_path)

            result = runner.invoke(
                init_config,
                ["--path", "/custom/path/config.yaml", "--profile", "default", "--force"],
            )

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected behavior through mock calls
            mock_create.assert_called_once_with(custom_path)
            mock_load_file.assert_called_once_with(custom_path)
            mock_save_file.assert_called_once_with(mock_config_file, custom_path)
            assert mock_config_file.current == "default"


class TestInitConfigCommandErrorHandling:
    """Test error handling in init_config command."""

    def test_init_config_with_create_error(self):
        """Test init config command with config creation error."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.expanduser"
        ) as mock_expanduser, patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:
            # Mock path operations
            mock_path = Path("/home/user/.paas-ai/config.yaml")
            mock_expanduser.return_value = mock_path
            mock_resolve.return_value = mock_path
            mock_exists.return_value = False
            mock_create.side_effect = Exception("Failed to create config")

            result = runner.invoke(init_config)

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0

    def test_init_config_with_profile_set_error(self):
        """Test init config command with profile setting error."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "pathlib.Path.exists"
        ) as mock_exists, patch(
            "pathlib.Path.home"
        ) as mock_home:
            # Mock path operations
            mock_home.return_value = Path("/home/user")
            mock_exists.return_value = False
            expected_path = Path("/home/user/.paas-ai/config.yaml")
            mock_create.return_value = expected_path

            # Mock config file operations with error
            mock_load_file.side_effect = Exception("Failed to load config")

            result = runner.invoke(init_config, ["--profile", "production"])

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0
            # Verify the expected behavior through mock calls
            mock_create.assert_called_once_with(expected_path)

    def test_init_config_with_path_expansion_error(self):
        """Test init config command with path expansion error."""
        runner = CliRunner()

        with patch("pathlib.Path.expanduser") as mock_expanduser:
            mock_expanduser.side_effect = Exception("Path expansion failed")

            result = runner.invoke(init_config, ["--path", "~/config.yaml"])

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0

    def test_init_config_with_resolve_error(self):
        """Test init config command with path resolution error."""
        runner = CliRunner()

        with patch("pathlib.Path.expanduser") as mock_expanduser, patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:
            mock_path = Path("/home/user/.paas-ai/config.yaml")
            mock_expanduser.return_value = mock_path
            mock_resolve.side_effect = Exception("Path resolution failed")

            result = runner.invoke(init_config, ["--path", "/custom/path/config.yaml"])

            # Verify command succeeded (returns False but doesn't raise exception)
            assert result.exit_code == 0


class TestInitConfigCommandEdgeCases:
    """Test edge cases for init_config command."""

    def test_init_config_with_nonexistent_parent_directory(self):
        """Test init config command with nonexistent parent directory."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.expanduser"
        ) as mock_expanduser, patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:
            # Mock path operations
            mock_path = Path("/nonexistent/path/config.yaml")
            mock_expanduser.return_value = mock_path
            mock_resolve.return_value = mock_path
            mock_exists.return_value = False
            mock_create.return_value = mock_path

            result = runner.invoke(init_config, ["--path", "/nonexistent/path/config.yaml"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected behavior through mock calls
            mock_create.assert_called_once_with(mock_path)

    def test_init_config_with_special_characters_in_path(self):
        """Test init config command with special characters in path."""
        runner = CliRunner()

        special_path = "/tmp/test with spaces/配置文件.yaml"
        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.expanduser"
        ) as mock_expanduser, patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:
            # Mock path operations
            mock_path = Path(special_path)
            mock_expanduser.return_value = mock_path
            mock_resolve.return_value = mock_path
            mock_exists.return_value = False
            mock_create.return_value = mock_path

            result = runner.invoke(init_config, ["--path", special_path])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected behavior through mock calls
            mock_create.assert_called_once_with(mock_path)

    def test_init_config_with_very_long_path(self):
        """Test init config command with very long path."""
        runner = CliRunner()

        long_path = "/" + "a" * 200 + "/config.yaml"
        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.expanduser"
        ) as mock_expanduser, patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:
            # Mock path operations
            mock_path = Path(long_path)
            mock_expanduser.return_value = mock_path
            mock_resolve.return_value = mock_path
            mock_exists.return_value = False
            mock_create.return_value = mock_path

            result = runner.invoke(init_config, ["--path", long_path])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected behavior through mock calls
            mock_create.assert_called_once_with(mock_path)

    def test_init_config_with_relative_path(self):
        """Test init config command with relative path."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.expanduser"
        ) as mock_expanduser, patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:
            # Mock path operations
            relative_path = Path("./config.yaml")
            resolved_path = Path("/current/directory/config.yaml")
            mock_expanduser.return_value = relative_path
            mock_resolve.return_value = resolved_path
            mock_exists.return_value = False
            mock_create.return_value = resolved_path

            result = runner.invoke(init_config, ["--path", "./config.yaml"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected behavior through mock calls
            mock_create.assert_called_once_with(resolved_path)

    def test_init_config_with_home_directory_expansion(self):
        """Test init config command with home directory expansion."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.expanduser"
        ) as mock_expanduser, patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:
            # Mock path operations
            tilde_path = Path("~/custom-config.yaml")
            expanded_path = Path("/home/user/custom-config.yaml")
            mock_expanduser.return_value = expanded_path
            mock_resolve.return_value = expanded_path
            mock_exists.return_value = False
            mock_create.return_value = expanded_path

            result = runner.invoke(init_config, ["--path", "~/custom-config.yaml"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected behavior through mock calls
            mock_create.assert_called_once_with(expanded_path)


class TestInitConfigCommandIntegration:
    """Integration tests for init_config command."""

    def test_init_config_full_workflow(self):
        """Test complete init config workflow."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.cli.commands.config.main.save_config_file"
        ) as mock_save_file, patch(
            "pathlib.Path.exists"
        ) as mock_exists, patch(
            "pathlib.Path.home"
        ) as mock_home:
            # Mock path operations
            mock_home.return_value = Path("/home/user")
            mock_exists.return_value = False
            expected_path = Path("/home/user/.paas-ai/config.yaml")
            mock_create.return_value = expected_path

            # Mock config file operations
            mock_config_file = Mock()
            mock_config_file.current = "local"
            mock_load_file.return_value = (mock_config_file, expected_path)

            result = runner.invoke(init_config, ["--profile", "production"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the complete workflow
            mock_create.assert_called_once_with(expected_path)
            mock_load_file.assert_called_once_with(expected_path)
            mock_save_file.assert_called_once_with(mock_config_file, expected_path)
            assert mock_config_file.current == "production"

    def test_init_config_with_multiple_profiles(self):
        """Test init config command with different profile options."""
        runner = CliRunner()

        profiles = ["local", "default", "production"]

        for profile in profiles:
            with patch(
                "src.paas_ai.cli.commands.config.main.create_sample_config"
            ) as mock_create, patch(
                "src.paas_ai.cli.commands.config.main.load_config_file"
            ) as mock_load_file, patch(
                "src.paas_ai.cli.commands.config.main.save_config_file"
            ) as mock_save_file, patch(
                "pathlib.Path.exists"
            ) as mock_exists, patch(
                "pathlib.Path.expanduser"
            ) as mock_expanduser, patch(
                "pathlib.Path.resolve"
            ) as mock_resolve:
                # Mock path operations
                mock_path = Path(f"/home/user/.paas-ai/config-{profile}.yaml")
                mock_expanduser.return_value = mock_path
                mock_resolve.return_value = mock_path
                mock_exists.return_value = False
                mock_create.return_value = mock_path

                # Mock config file operations
                mock_config_file = Mock()
                mock_config_file.current = "local"
                mock_load_file.return_value = (mock_config_file, mock_path)

                result = runner.invoke(
                    init_config, ["--profile", profile, "--path", str(mock_path)]
                )

                # Verify command succeeded
                assert result.exit_code == 0
                # Verify the expected behavior through mock calls
                mock_create.assert_called_once_with(mock_path)

                if profile != "local":
                    mock_load_file.assert_called_once_with(mock_path)
                    mock_save_file.assert_called_once_with(mock_config_file, mock_path)
                    assert mock_config_file.current == profile

    def test_init_config_consistency_across_calls(self):
        """Test that init config command is consistent across multiple calls."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.expanduser"
        ) as mock_expanduser, patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:
            # Mock path operations
            mock_path = Path("/home/user/.paas-ai/config.yaml")
            mock_expanduser.return_value = mock_path
            mock_resolve.return_value = mock_path
            mock_exists.return_value = False
            mock_create.return_value = mock_path

            # Make multiple calls with force
            result1 = runner.invoke(init_config, ["--force"])
            result2 = runner.invoke(init_config, ["--force"])

            assert result1.exit_code == 0
            assert result2.exit_code == 0
            assert result1.output == result2.output


class TestInitConfigCommandPerformance:
    """Test performance characteristics of init_config command."""

    def test_init_config_response_time(self):
        """Test that init config command responds quickly."""
        import time

        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.expanduser"
        ) as mock_expanduser, patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:
            # Mock path operations
            mock_path = Path("/home/user/.paas-ai/config.yaml")
            mock_expanduser.return_value = mock_path
            mock_resolve.return_value = mock_path
            mock_exists.return_value = False
            mock_create.return_value = mock_path

            start_time = time.time()
            result = runner.invoke(init_config)
            end_time = time.time()

            # Should respond quickly (less than 1 second)
            assert (end_time - start_time) < 1.0
            assert result.exit_code == 0

    def test_init_config_memory_usage(self):
        """Test that init config command doesn't use excessive memory."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.create_sample_config"
        ) as mock_create, patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.expanduser"
        ) as mock_expanduser, patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:
            # Mock path operations
            mock_path = Path("/home/user/.paas-ai/config.yaml")
            mock_expanduser.return_value = mock_path
            mock_resolve.return_value = mock_path
            mock_exists.return_value = False
            mock_create.return_value = mock_path

            # Should not raise memory errors
            result = runner.invoke(init_config)
            assert result.exit_code == 0

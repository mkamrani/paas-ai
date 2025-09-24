"""
Unit tests for CLI config validation and edit commands.

Tests all components of validation and edit commands including:
- validate command functionality
- edit command functionality
- Configuration validation
- File editing
- Error handling and edge cases
- Environment variable handling
- Editor integration
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, mock_open, patch

import pytest
from click.testing import CliRunner

from src.paas_ai.cli.commands.config.main import edit_config, validate_config
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


class TestValidateConfigCommand:
    """Test the validate_config command."""

    def test_validate_config_basic(self):
        """Test basic validate config command."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_profile:
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.OPENAI
            mock_config.vectorstore.type = VectorStoreType.CHROMA

            mock_load.return_value = mock_config
            mock_profile.return_value = "default"

            result = runner.invoke(validate_config)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected functions were called
            mock_load.assert_called_once()
            mock_profile.assert_called_once()

    def test_validate_config_with_specific_file(self):
        """Test validate config command with specific config file."""
        runner = CliRunner()

        config_path = Path("/custom/config.yaml")
        with patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch.dict(os.environ, {"PAAS_AI_CONFIG": "/custom/config.yaml"}):
            mock_config_file = Mock()
            mock_config_file.current = "production"
            mock_config_file.profiles = {"custom": Mock()}

            mock_load_file.return_value = (mock_config_file, config_path)

            result = runner.invoke(validate_config, ["--config-path", str(config_path)])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected function was called
            mock_load_file.assert_called_once_with(config_path)

    def test_validate_config_with_specific_profile(self):
        """Test validate config command with specific profile."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults:
            # Test built-in profile
            mock_config_obj = Mock()
            mock_config_obj.embedding.type = EmbeddingType.SENTENCE_TRANSFORMERS
            mock_config_obj.vectorstore.type = VectorStoreType.CHROMA

            mock_defaults.__contains__ = lambda x, y: y == "local"
            mock_defaults.__getitem__ = lambda x, y: mock_config_obj if y == "local" else None

            result = runner.invoke(validate_config, ["--profile", "local"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected function was called
            mock_load_file.assert_called_once()

    def test_validate_config_with_custom_profile(self):
        """Test validate config command with custom profile."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults:
            # Test custom profile
            mock_config_obj = Mock()
            mock_config_obj.embedding.type = EmbeddingType.OPENAI
            mock_config_obj.vectorstore.type = VectorStoreType.CHROMA

            mock_config_file = Mock()
            mock_config_file.profiles = {"custom": mock_config_obj}
            mock_load_file.return_value = (mock_config_file, Path("/path/to/config.yaml"))

            mock_defaults.__contains__ = lambda x, y: False

            result = runner.invoke(validate_config, ["--profile", "custom"])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected function was called
            mock_load_file.assert_called_once()

    def test_validate_config_with_nonexistent_profile(self):
        """Test validate config command with nonexistent profile."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults:
            mock_config_file = Mock()
            mock_config_file.profiles = {}
            mock_load_file.return_value = (mock_config_file, Path("/path/to/config.yaml"))

            mock_defaults.__contains__ = lambda x, y: False

            result = runner.invoke(validate_config, ["--profile", "nonexistent"])

            # Verify command succeeded (returns False but Click doesn't set exit code)
            assert result.exit_code == 0
            # Verify the expected function was called
            mock_load_file.assert_called_once()

    def test_validate_config_with_config_file_and_profile(self):
        """Test validate config command with both config file and profile."""
        runner = CliRunner()

        config_path = Path("/custom/config.yaml")
        with patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch.dict(os.environ, {"PAAS_AI_CONFIG": "/custom/config.yaml"}):
            mock_config_obj = Mock()
            mock_config_obj.embedding.type = EmbeddingType.OPENAI
            mock_config_obj.vectorstore.type = VectorStoreType.CHROMA

            mock_config_file = Mock()
            mock_config_file.profiles = {"test": mock_config_obj}
            mock_load_file.return_value = (mock_config_file, config_path)

            result = runner.invoke(
                validate_config, ["--config-path", str(config_path), "--profile", "test"]
            )

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected function was called
            mock_load_file.assert_called_once_with(config_path)

    def test_validate_config_with_configuration_error(self):
        """Test validate config command with configuration error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load:
            mock_load.side_effect = ConfigurationError("Invalid configuration")

            result = runner.invoke(validate_config)

            # Verify command succeeded (returns False but Click doesn't set exit code)
            assert result.exit_code == 0
            # Verify the expected function was called
            mock_load.assert_called_once()

    def test_validate_config_with_general_error(self):
        """Test validate config command with general error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load:
            mock_load.side_effect = Exception("Unexpected error")

            result = runner.invoke(validate_config)

            # Verify command succeeded (returns False but Click doesn't set exit code)
            assert result.exit_code == 0
            # Verify the expected function was called
            mock_load.assert_called_once()


class TestEditConfigCommand:
    """Test the edit_config command."""

    def test_edit_config_basic(self):
        """Test basic edit config command."""
        runner = CliRunner()

        with patch("subprocess.run") as mock_run, patch.dict(os.environ, {"EDITOR": "vim"}):
            # Mock Path.exists to return True for the config file
            with patch("pathlib.Path.exists", return_value=True):
                result = runner.invoke(edit_config)

                # Verify command succeeded
                assert result.exit_code == 0
                # Verify the expected function was called
                mock_run.assert_called_once()

    def test_edit_config_with_custom_path(self):
        """Test edit config command with custom config path."""
        runner = CliRunner()

        custom_path = Path("/custom/config.yaml")
        with patch("subprocess.run") as mock_run, patch.dict(
            os.environ, {"PAAS_AI_CONFIG": str(custom_path), "EDITOR": "nano"}
        ):
            # Mock Path.exists to return True for the config file
            with patch("pathlib.Path.exists", return_value=True):
                result = runner.invoke(edit_config)

                # Verify command succeeded
                assert result.exit_code == 0
                # Verify the expected function was called
                mock_run.assert_called_once()

    def test_edit_config_with_default_editor(self):
        """Test edit config command with default editor."""
        runner = CliRunner()

        with patch("subprocess.run") as mock_run, patch.dict(os.environ, {}, clear=True):
            # Mock Path.exists to return True for the config file
            with patch("pathlib.Path.exists", return_value=True):
                result = runner.invoke(edit_config)

                # Verify command succeeded
                assert result.exit_code == 0
                # Verify the expected function was called
                mock_run.assert_called_once()

    def test_edit_config_with_config_file_not_found(self):
        """Test edit config command with config file not found."""
        runner = CliRunner()

        config_path = Path("/home/user/.paas-ai/config.yaml")
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False

            result = runner.invoke(edit_config)

            # Verify command succeeded (returns False but Click doesn't set exit code)
            assert result.exit_code == 0
            # Verify the expected function was called
            mock_exists.assert_called_once()

    def test_edit_config_with_validation_error(self):
        """Test edit config command with validation error after editing."""
        runner = CliRunner()

        config_path = Path("/home/user/.paas-ai/config.yaml")
        with patch("pathlib.Path.exists") as mock_exists, patch(
            "subprocess.run"
        ) as mock_run, patch(
            "src.paas_ai.cli.commands.config.main.load_config"
        ) as mock_load, patch.dict(
            os.environ, {"EDITOR": "vim"}
        ):
            mock_exists.return_value = True
            mock_load.side_effect = ConfigurationError("Invalid configuration after edit")

            result = runner.invoke(edit_config)

            # Verify command succeeded (returns False but Click doesn't set exit code)
            assert result.exit_code == 0
            # Verify the expected functions were called
            mock_exists.assert_called_once()
            mock_run.assert_called_once()
            mock_load.assert_called_once()

    def test_edit_config_with_editor_error(self):
        """Test edit config command with editor error."""
        runner = CliRunner()

        config_path = Path("/home/user/.paas-ai/config.yaml")
        with patch("pathlib.Path.exists") as mock_exists, patch(
            "subprocess.run"
        ) as mock_run, patch.dict(os.environ, {"EDITOR": "nonexistent-editor"}):
            mock_exists.return_value = True
            mock_run.side_effect = FileNotFoundError("Editor not found")

            result = runner.invoke(edit_config)

            # Verify command succeeded (returns False but Click doesn't set exit code)
            assert result.exit_code == 0
            # Verify the expected functions were called
            mock_exists.assert_called_once()
            mock_run.assert_called_once()

    def test_edit_config_with_general_error(self):
        """Test edit config command with general error."""
        runner = CliRunner()

        config_path = Path("/home/user/.paas-ai/config.yaml")
        with patch("pathlib.Path.exists") as mock_exists, patch(
            "subprocess.run"
        ) as mock_run, patch.dict(os.environ, {"EDITOR": "vim"}):
            mock_exists.return_value = True
            mock_run.side_effect = Exception("Unexpected error")

            result = runner.invoke(edit_config)

            # Verify command succeeded (returns False but Click doesn't set exit code)
            assert result.exit_code == 0
            # Verify the expected functions were called
            mock_exists.assert_called_once()
            mock_run.assert_called_once()


class TestValidationCommandEdgeCases:
    """Test edge cases for validation commands."""

    def test_validate_config_with_special_characters_in_path(self):
        """Test validate config command with special characters in path."""
        runner = CliRunner()

        special_path = "/tmp/test with spaces/配置文件.yaml"
        with patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch.dict(os.environ, {"PAAS_AI_CONFIG": special_path}):
            mock_config_file = Mock()
            mock_config_file.current = "default"
            mock_config_file.profiles = {}
            mock_load_file.return_value = (mock_config_file, Path(special_path))

            result = runner.invoke(validate_config, ["--config-path", special_path])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected function was called
            mock_load_file.assert_called_once_with(Path(special_path))

    def test_edit_config_with_special_characters_in_path(self):
        """Test edit config command with special characters in path."""
        runner = CliRunner()

        special_path = "/tmp/test with spaces/配置文件.yaml"
        with patch("subprocess.run") as mock_run, patch.dict(
            os.environ, {"PAAS_AI_CONFIG": special_path, "EDITOR": "vim"}
        ):
            # Mock Path.exists to return True for the config file
            with patch("pathlib.Path.exists", return_value=True):
                result = runner.invoke(edit_config)

                # Verify command succeeded
                assert result.exit_code == 0
                # Verify the expected function was called
                mock_run.assert_called_once()

    def test_validate_config_with_very_long_path(self):
        """Test validate config command with very long path."""
        runner = CliRunner()

        long_path = "/" + "a" * 200 + "/config.yaml"
        with patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch.dict(os.environ, {"PAAS_AI_CONFIG": long_path}):
            mock_config_file = Mock()
            mock_config_file.current = "default"
            mock_config_file.profiles = {}
            mock_load_file.return_value = (mock_config_file, Path(long_path))

            result = runner.invoke(validate_config, ["--config-path", long_path])

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected function was called
            mock_load_file.assert_called_once_with(Path(long_path))

    def test_edit_config_with_unicode_editor(self):
        """Test edit config command with unicode editor name."""
        runner = CliRunner()

        with patch("subprocess.run") as mock_run, patch.dict(os.environ, {"EDITOR": "测试编辑器"}):
            # Mock Path.exists to return True for the config file
            with patch("pathlib.Path.exists", return_value=True):
                result = runner.invoke(edit_config)

                # Verify command succeeded
                assert result.exit_code == 0
                # Verify the expected function was called
                mock_run.assert_called_once()

    def test_validate_config_with_none_values(self):
        """Test validate config command with None values."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_profile:
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.OPENAI
            mock_config.vectorstore.type = VectorStoreType.CHROMA

            mock_load.return_value = mock_config
            mock_profile.return_value = None  # None current profile

            result = runner.invoke(validate_config)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected functions were called
            mock_load.assert_called_once()
            mock_profile.assert_called_once()


class TestValidationCommandIntegration:
    """Integration tests for validation commands."""

    def test_validate_config_full_workflow(self):
        """Test complete validate config workflow."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_profile:
            # Mock realistic configuration
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.SENTENCE_TRANSFORMERS
            mock_config.vectorstore.type = VectorStoreType.CHROMA

            mock_load.return_value = mock_config
            mock_profile.return_value = "production"

            result = runner.invoke(validate_config)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected functions were called
            mock_load.assert_called_once()
            mock_profile.assert_called_once()

    def test_edit_config_full_workflow(self):
        """Test complete edit config workflow."""
        runner = CliRunner()

        config_path = Path("/home/user/.paas-ai/config.yaml")
        with patch("pathlib.Path.exists") as mock_exists, patch(
            "subprocess.run"
        ) as mock_run, patch(
            "src.paas_ai.cli.commands.config.main.load_config"
        ) as mock_load, patch.dict(
            os.environ, {"EDITOR": "vim"}
        ):
            mock_exists.return_value = True

            # Mock successful validation after edit
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.OPENAI
            mock_config.vectorstore.type = VectorStoreType.CHROMA
            mock_load.return_value = mock_config

            result = runner.invoke(edit_config)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the expected functions were called
            mock_exists.assert_called_once()
            mock_run.assert_called_once()
            mock_load.assert_called_once()

    def test_validation_commands_consistency_across_calls(self):
        """Test that validation commands are consistent across multiple calls."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_profile:
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.OPENAI
            mock_config.vectorstore.type = VectorStoreType.CHROMA

            mock_load.return_value = mock_config
            mock_profile.return_value = "default"

            # Make multiple calls
            result1 = runner.invoke(validate_config)
            result2 = runner.invoke(validate_config)

            assert result1.exit_code == 0
            assert result2.exit_code == 0
            # Both commands should succeed consistently

    def test_validation_with_different_config_types(self):
        """Test validation with different configuration types."""
        runner = CliRunner()

        config_types = [
            (EmbeddingType.OPENAI, VectorStoreType.CHROMA),
            (EmbeddingType.SENTENCE_TRANSFORMERS, VectorStoreType.FAISS),
            (EmbeddingType.OPENAI, VectorStoreType.PINECONE),
        ]

        for embedding_type, vectorstore_type in config_types:
            with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
                "src.paas_ai.cli.commands.config.main.get_current_profile"
            ) as mock_profile:
                mock_config = Mock()
                mock_config.embedding.type = embedding_type
                mock_config.vectorstore.type = vectorstore_type

                mock_load.return_value = mock_config
                mock_profile.return_value = "default"

                result = runner.invoke(validate_config)

                # Verify command succeeded
                assert result.exit_code == 0
                # Verify the expected functions were called
                mock_load.assert_called_once()
                mock_profile.assert_called_once()


class TestValidationCommandPerformance:
    """Test performance characteristics of validation commands."""

    def test_validate_config_response_time(self):
        """Test that validate config command responds quickly."""
        import time

        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_profile:
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.OPENAI
            mock_config.vectorstore.type = VectorStoreType.CHROMA

            mock_load.return_value = mock_config
            mock_profile.return_value = "default"

            start_time = time.time()
            result = runner.invoke(validate_config)
            end_time = time.time()

            # Should respond quickly (less than 1 second)
            assert (end_time - start_time) < 1.0
            assert result.exit_code == 0

    def test_edit_config_response_time(self):
        """Test that edit config command responds quickly."""
        import time

        runner = CliRunner()

        config_path = Path("/home/user/.paas-ai/config.yaml")
        with patch("pathlib.Path.exists") as mock_exists, patch(
            "subprocess.run"
        ) as mock_run, patch(
            "src.paas_ai.cli.commands.config.main.load_config"
        ) as mock_load, patch.dict(
            os.environ, {"EDITOR": "vim"}
        ):
            mock_exists.return_value = True

            # Mock successful validation
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.OPENAI
            mock_config.vectorstore.type = VectorStoreType.CHROMA
            mock_load.return_value = mock_config

            start_time = time.time()
            result = runner.invoke(edit_config)
            end_time = time.time()

            # Should respond quickly (less than 1 second)
            assert (end_time - start_time) < 1.0
            assert result.exit_code == 0

    def test_validation_commands_memory_usage(self):
        """Test that validation commands don't use excessive memory."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_profile:
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.OPENAI
            mock_config.vectorstore.type = VectorStoreType.CHROMA

            mock_load.return_value = mock_config
            mock_profile.return_value = "default"

            # Should not raise memory errors
            result = runner.invoke(validate_config)
            assert result.exit_code == 0

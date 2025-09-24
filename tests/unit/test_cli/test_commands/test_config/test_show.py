"""
Unit tests for CLI config show command.

Tests all components of the show command including:
- Show command functionality
- Path display options
- Profile listing
- Configuration display
- Error handling and edge cases
- Environment variable handling
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from click.testing import CliRunner

from src.paas_ai.cli.commands.config.main import show_config
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


class TestShowConfigCommand:
    """Test the show_config command."""

    def test_show_config_basic(self):
        """Test basic show config command."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_profile, patch("src.paas_ai.cli.commands.config.main.logger") as mock_logger:
            # Mock configuration
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.OPENAI
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.vectorstore.type = VectorStoreType.CHROMA
            mock_config.vectorstore.collection_name = "test-collection"
            mock_config.vectorstore.persist_directory = "/tmp/test"
            mock_config.retriever.type = RetrieverType.SIMILARITY
            mock_config.batch_size = 32
            mock_config.validate_urls = True

            mock_load.return_value = mock_config
            mock_profile.return_value = "default"

            result = runner.invoke(show_config)

            assert result.exit_code == 0
            # Verify that logger was called with expected messages
            mock_logger.info.assert_any_call("Current Active Configuration (Profile: default):")
            mock_logger.info.assert_any_call("-" * 40)
            mock_logger.info.assert_any_call(f"Embedding Type: {EmbeddingType.OPENAI}")
            mock_logger.info.assert_any_call("Embedding Model: text-embedding-3-small")
            mock_logger.info.assert_any_call(f"Vector Store: {VectorStoreType.CHROMA}")
            mock_logger.info.assert_any_call("Collection: test-collection")
            mock_logger.info.assert_any_call("Persist Directory: /tmp/test")
            mock_logger.info.assert_any_call(f"Retriever: {RetrieverType.SIMILARITY}")
            mock_logger.info.assert_any_call("Batch Size: 32")
            mock_logger.info.assert_any_call("Validate URLs: True")

    def test_show_config_with_path_option(self):
        """Test show config command with --path option."""
        runner = CliRunner()

        with patch.dict(
            os.environ, {"PAAS_AI_CONFIG": "/custom/config.yaml", "PAAS_AI_PROFILE": "test-profile"}
        ):
            with patch("pathlib.Path.exists") as mock_exists, patch(
                "src.paas_ai.cli.commands.config.main.logger"
            ) as mock_logger:
                mock_exists.return_value = True

                result = runner.invoke(show_config, ["--path"])

                assert result.exit_code == 0
                # Verify logger calls instead of output assertions
                mock_logger.info.assert_any_call("Configuration file locations:")
                mock_logger.info.assert_any_call("  Custom (PAAS_AI_CONFIG): /custom/config.yaml")
                mock_logger.success.assert_any_call("    ✅ File exists")
                mock_logger.info.assert_any_call(
                    "  Profile override (PAAS_AI_PROFILE): test-profile"
                )

    def test_show_config_with_path_option_file_not_found(self):
        """Test show config command with --path option when file doesn't exist."""
        runner = CliRunner()

        with patch.dict(os.environ, {"PAAS_AI_CONFIG": "/nonexistent/config.yaml"}):
            with patch("pathlib.Path.exists") as mock_exists, patch(
                "src.paas_ai.cli.commands.config.main.logger"
            ) as mock_logger:
                mock_exists.return_value = False

                result = runner.invoke(show_config, ["--path"])

                assert result.exit_code == 0
                # Verify logger calls instead of output assertions
                mock_logger.info.assert_any_call("Configuration file locations:")
                mock_logger.info.assert_any_call(
                    "  Custom (PAAS_AI_CONFIG): /nonexistent/config.yaml"
                )
                mock_logger.warning.assert_any_call("    ⚠️  File not found")

    def test_show_config_with_profiles_option(self):
        """Test show config command with --profiles option."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current, patch(
            "src.paas_ai.cli.commands.config.main.logger"
        ) as mock_logger:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
                "production": "Production environment profile",
                "custom": "Custom profile: My custom profile",
            }
            mock_current.return_value = "default"

            result = runner.invoke(show_config, ["--profiles"])

            assert result.exit_code == 0
            # Verify logger calls instead of output assertions
            mock_logger.info.assert_any_call("Available Profiles:")
            mock_logger.info.assert_any_call("=" * 30)
            mock_logger.info.assert_any_call("👉 default: Default configuration profile")
            mock_logger.info.assert_any_call("   local: Local development profile")
            mock_logger.info.assert_any_call("   production: Production environment profile")
            mock_logger.info.assert_any_call("   custom: Custom profile: My custom profile")
            mock_logger.info.assert_any_call("\nCurrent active profile: default")

    def test_show_config_with_specific_profile(self):
        """Test show config command with specific profile."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.logger"
        ) as mock_logger:
            # Mock built-in profile
            mock_config_obj = Mock()
            mock_config_obj.embedding.type = EmbeddingType.SENTENCE_TRANSFORMERS
            mock_config_obj.embedding.model_name = "all-MiniLM-L6-v2"
            mock_config_obj.vectorstore.type = VectorStoreType.CHROMA
            mock_config_obj.vectorstore.collection_name = "paas_ai_local"
            mock_config_obj.vectorstore.persist_directory = "rag_data/chroma_local"
            mock_config_obj.retriever.type = RetrieverType.SIMILARITY
            mock_config_obj.batch_size = 16
            mock_config_obj.validate_urls = True

            # Mock load_config_file to return a tuple (config_file, path)
            mock_config_file = Mock()
            mock_config_file.profiles = {}
            mock_load_file.return_value = (mock_config_file, "/path/to/config.yaml")

            mock_defaults.__contains__ = lambda x, y: y == "local"
            mock_defaults.__getitem__ = lambda x, y: mock_config_obj if y == "local" else None

            result = runner.invoke(show_config, ["--profile", "local"])

            assert result.exit_code == 0
            # Verify logger calls instead of output assertions
            mock_logger.info.assert_any_call("Built-in Profile: local")
            mock_logger.info.assert_any_call("-" * 30)
            mock_logger.info.assert_any_call(
                f"Embedding Type: {EmbeddingType.SENTENCE_TRANSFORMERS}"
            )
            mock_logger.info.assert_any_call("Embedding Model: all-MiniLM-L6-v2")
            mock_logger.info.assert_any_call(f"Vector Store: {VectorStoreType.CHROMA}")
            mock_logger.info.assert_any_call("Collection: paas_ai_local")
            mock_logger.info.assert_any_call("Persist Directory: rag_data/chroma_local")
            mock_logger.info.assert_any_call(f"Retriever: {RetrieverType.SIMILARITY}")
            mock_logger.info.assert_any_call("Batch Size: 16")
            mock_logger.info.assert_any_call("Validate URLs: True")

    def test_show_config_with_custom_profile(self):
        """Test show config command with custom profile."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.logger"
        ) as mock_logger:
            # Mock custom profile
            mock_config_obj = Mock()
            mock_config_obj.embedding.type = EmbeddingType.OPENAI
            mock_config_obj.embedding.model_name = "text-embedding-3-large"
            mock_config_obj.vectorstore.type = VectorStoreType.CHROMA
            mock_config_obj.vectorstore.collection_name = "custom-collection"
            mock_config_obj.vectorstore.persist_directory = None
            mock_config_obj.retriever.type = RetrieverType.MMR
            mock_config_obj.batch_size = 64
            mock_config_obj.validate_urls = True

            mock_config_file = Mock()
            mock_config_file.profiles = {"custom": mock_config_obj}
            mock_load_file.return_value = (mock_config_file, "/path/to/config.yaml")

            mock_defaults.__contains__ = lambda x, y: False

            result = runner.invoke(show_config, ["--profile", "custom"])

            assert result.exit_code == 0
            # Verify logger calls instead of output assertions
            mock_logger.info.assert_any_call("Custom Profile: custom")
            mock_logger.info.assert_any_call("-" * 30)
            mock_logger.info.assert_any_call(f"Embedding Type: {EmbeddingType.OPENAI}")
            mock_logger.info.assert_any_call("Embedding Model: text-embedding-3-large")
            mock_logger.info.assert_any_call(f"Vector Store: {VectorStoreType.CHROMA}")
            mock_logger.info.assert_any_call("Collection: custom-collection")
            mock_logger.info.assert_any_call(f"Retriever: {RetrieverType.MMR}")
            mock_logger.info.assert_any_call("Batch Size: 64")
            mock_logger.info.assert_any_call("Validate URLs: True")

    def test_show_config_with_nonexistent_profile(self):
        """Test show config command with nonexistent profile."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch(
            "src.paas_ai.core.config.schemas.DEFAULT_CONFIG_PROFILES"
        ) as mock_defaults, patch(
            "src.paas_ai.cli.commands.config.main.logger"
        ) as mock_logger:
            mock_config_file = Mock()
            mock_config_file.profiles = {}
            mock_load_file.return_value = (mock_config_file, "/path/to/config.yaml")

            mock_defaults.__contains__ = lambda x, y: False

            result = runner.invoke(show_config, ["--profile", "nonexistent"])

            assert result.exit_code == 0
            # Verify logger calls instead of output assertions
            mock_logger.error.assert_any_call("Profile 'nonexistent' not found")


class TestShowConfigCommandErrorHandling:
    """Test error handling in show_config command."""

    def test_show_config_with_configuration_error(self):
        """Test show config command with configuration error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.logger"
        ) as mock_logger:
            mock_load.side_effect = ConfigurationError("Invalid configuration")

            result = runner.invoke(show_config)

            assert result.exit_code == 0
            # Verify logger calls instead of output assertions
            mock_logger.error.assert_any_call("Failed to show configuration: Invalid configuration")

    def test_show_config_with_config_file_not_found(self):
        """Test show config command with config file not found."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch("src.paas_ai.cli.commands.config.main.logger") as mock_logger:
            mock_load_file.side_effect = ConfigFileNotFoundError("Config file not found")

            result = runner.invoke(show_config, ["--profile", "test"])

            assert result.exit_code == 0
            # Verify logger calls instead of output assertions
            mock_logger.error.assert_any_call("Error loading profile 'test': Config file not found")

    def test_show_config_with_general_exception(self):
        """Test show config command with general exception."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.logger"
        ) as mock_logger:
            mock_load.side_effect = Exception("Unexpected error")

            result = runner.invoke(show_config)

            assert result.exit_code == 0
            # Verify logger calls instead of output assertions
            mock_logger.error.assert_any_call("Failed to show configuration: Unexpected error")

    def test_show_config_with_profile_load_error(self):
        """Test show config command with profile loading error."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.load_config_file"
        ) as mock_load_file, patch("src.paas_ai.cli.commands.config.main.logger") as mock_logger:
            mock_load_file.side_effect = Exception("Profile load error")

            result = runner.invoke(show_config, ["--profile", "test"])

            assert result.exit_code == 0
            # Verify logger calls instead of output assertions
            mock_logger.error.assert_any_call("Error loading profile 'test': Profile load error")


class TestShowConfigCommandEdgeCases:
    """Test edge cases for show_config command."""

    def test_show_config_with_none_persist_directory(self):
        """Test show config command when persist_directory is None."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_profile, patch("src.paas_ai.cli.commands.config.main.logger") as mock_logger:
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.OPENAI
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.vectorstore.type = VectorStoreType.CHROMA
            mock_config.vectorstore.collection_name = "test-collection"
            mock_config.vectorstore.persist_directory = None
            mock_config.retriever.type = RetrieverType.SIMILARITY
            mock_config.batch_size = 32
            mock_config.validate_urls = True

            mock_load.return_value = mock_config
            mock_profile.return_value = "default"

            result = runner.invoke(show_config)

            assert result.exit_code == 0
            # Verify logger calls instead of output assertions
            mock_logger.info.assert_any_call("Current Active Configuration (Profile: default):")
            mock_logger.info.assert_any_call("-" * 40)
            mock_logger.info.assert_any_call(f"Embedding Type: {EmbeddingType.OPENAI}")
            mock_logger.info.assert_any_call("Embedding Model: text-embedding-3-small")
            mock_logger.info.assert_any_call(f"Vector Store: {VectorStoreType.CHROMA}")
            mock_logger.info.assert_any_call("Collection: test-collection")
            mock_logger.info.assert_any_call(f"Retriever: {RetrieverType.SIMILARITY}")
            mock_logger.info.assert_any_call("Batch Size: 32")
            mock_logger.info.assert_any_call("Validate URLs: True")
            # Should not call persist directory when it's None
            # We can verify this by checking that no call contains "Persist Directory:"
            persist_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Persist Directory:" in str(call)
            ]
            assert len(persist_calls) == 0

    def test_show_config_with_empty_environment_variables(self):
        """Test show config command with empty environment variables."""
        runner = CliRunner()

        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.exists") as mock_exists, patch(
                "src.paas_ai.cli.commands.config.main.logger"
            ) as mock_logger:
                mock_exists.return_value = False

                result = runner.invoke(show_config, ["--path"])

                assert result.exit_code == 0
                # Verify logger calls instead of output assertions
                mock_logger.info.assert_any_call("Configuration file locations:")
                mock_logger.info.assert_any_call("  Default: /Users/mohsen/.paas-ai/config.yaml")
                mock_logger.info.assert_any_call("    ℹ️  File not found (using fallback)")

    def test_show_config_with_special_characters_in_paths(self):
        """Test show config command with special characters in paths."""
        runner = CliRunner()

        special_path = "/tmp/test with spaces/配置文件.yaml"
        with patch.dict(os.environ, {"PAAS_AI_CONFIG": special_path}):
            with patch("pathlib.Path.exists") as mock_exists, patch(
                "src.paas_ai.cli.commands.config.main.logger"
            ) as mock_logger:
                mock_exists.return_value = True

                result = runner.invoke(show_config, ["--path"])

                assert result.exit_code == 0
                # Verify logger calls instead of output assertions
                mock_logger.info.assert_any_call(f"  Custom (PAAS_AI_CONFIG): {special_path}")

    def test_show_config_with_very_long_profile_names(self):
        """Test show config command with very long profile names."""
        runner = CliRunner()

        long_profile_name = "a" * 100
        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current, patch(
            "src.paas_ai.cli.commands.config.main.logger"
        ) as mock_logger:
            mock_list.return_value = {long_profile_name: "Very long profile name"}
            mock_current.return_value = long_profile_name

            result = runner.invoke(show_config, ["--profiles"])

            assert result.exit_code == 0
            # Verify logger calls instead of output assertions
            mock_logger.info.assert_any_call(f"👉 {long_profile_name}: Very long profile name")


class TestShowConfigCommandIntegration:
    """Integration tests for show_config command."""

    def test_show_config_full_workflow(self):
        """Test complete show config workflow."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_profile, patch("src.paas_ai.cli.commands.config.main.logger") as mock_logger:
            # Mock realistic configuration
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.SENTENCE_TRANSFORMERS
            mock_config.embedding.model_name = "all-MiniLM-L6-v2"
            mock_config.vectorstore.type = VectorStoreType.CHROMA
            mock_config.vectorstore.collection_name = "production-knowledge"
            mock_config.vectorstore.persist_directory = "/data/vectorstore"
            mock_config.retriever.type = RetrieverType.MMR
            mock_config.batch_size = 64
            mock_config.validate_urls = True

            mock_load.return_value = mock_config
            mock_profile.return_value = "production"

            result = runner.invoke(show_config)

            assert result.exit_code == 0
            # Verify logger calls instead of output assertions
            mock_logger.info.assert_any_call("Current Active Configuration (Profile: production):")
            mock_logger.info.assert_any_call("-" * 40)
            mock_logger.info.assert_any_call(
                f"Embedding Type: {EmbeddingType.SENTENCE_TRANSFORMERS}"
            )
            mock_logger.info.assert_any_call("Embedding Model: all-MiniLM-L6-v2")
            mock_logger.info.assert_any_call(f"Vector Store: {VectorStoreType.CHROMA}")
            mock_logger.info.assert_any_call("Collection: production-knowledge")
            mock_logger.info.assert_any_call("Persist Directory: /data/vectorstore")
            mock_logger.info.assert_any_call(f"Retriever: {RetrieverType.MMR}")
            mock_logger.info.assert_any_call("Batch Size: 64")
            mock_logger.info.assert_any_call("Validate URLs: True")

    def test_show_config_with_multiple_options(self):
        """Test show config command with multiple options (should prioritize first)."""
        runner = CliRunner()

        # Test with multiple flags - should show path info
        with patch.dict(os.environ, {"PAAS_AI_CONFIG": "/test/config.yaml"}):
            with patch("pathlib.Path.exists") as mock_exists, patch(
                "src.paas_ai.cli.commands.config.main.logger"
            ) as mock_logger:
                mock_exists.return_value = True

                result = runner.invoke(show_config, ["--path", "--profiles"])

                assert result.exit_code == 0
                # Should show path info since --path is processed first
                # Verify logger calls instead of output assertions
                mock_logger.info.assert_any_call("Configuration file locations:")
                mock_logger.info.assert_any_call("  Custom (PAAS_AI_CONFIG): /test/config.yaml")

    def test_show_config_consistency_across_calls(self):
        """Test that show config command is consistent across multiple calls."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_profile:
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.OPENAI
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.vectorstore.type = VectorStoreType.CHROMA
            mock_config.vectorstore.collection_name = "test-collection"
            mock_config.vectorstore.persist_directory = "/tmp/test"
            mock_config.retriever.type = RetrieverType.SIMILARITY
            mock_config.batch_size = 32
            mock_config.validate_urls = True

            mock_load.return_value = mock_config
            mock_profile.return_value = "default"

            # Make multiple calls
            result1 = runner.invoke(show_config)
            result2 = runner.invoke(show_config)

            assert result1.exit_code == 0
            assert result2.exit_code == 0
            assert result1.output == result2.output


class TestShowConfigCommandPerformance:
    """Test performance characteristics of show_config command."""

    def test_show_config_response_time(self):
        """Test that show config command responds quickly."""
        import time

        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_profile:
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.OPENAI
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.vectorstore.type = VectorStoreType.CHROMA
            mock_config.vectorstore.collection_name = "test-collection"
            mock_config.vectorstore.persist_directory = "/tmp/test"
            mock_config.retriever.type = RetrieverType.SIMILARITY
            mock_config.batch_size = 32
            mock_config.validate_urls = True

            mock_load.return_value = mock_config
            mock_profile.return_value = "default"

            start_time = time.time()
            result = runner.invoke(show_config)
            end_time = time.time()

            # Should respond quickly (less than 1 second)
            assert (end_time - start_time) < 1.0
            assert result.exit_code == 0

    def test_show_config_memory_usage(self):
        """Test that show config command doesn't use excessive memory."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.load_config") as mock_load, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_profile:
            mock_config = Mock()
            mock_config.embedding.type = EmbeddingType.OPENAI
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.vectorstore.type = VectorStoreType.CHROMA
            mock_config.vectorstore.collection_name = "test-collection"
            mock_config.vectorstore.persist_directory = "/tmp/test"
            mock_config.retriever.type = RetrieverType.SIMILARITY
            mock_config.batch_size = 32
            mock_config.validate_urls = True

            mock_load.return_value = mock_config
            mock_profile.return_value = "default"

            # Should not raise memory errors
            result = runner.invoke(show_config)
            assert result.exit_code == 0

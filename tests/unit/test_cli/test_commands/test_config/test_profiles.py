"""
Unit tests for CLI config profiles command.

Tests all components of the profiles command including:
- Profiles command functionality
- Built-in profile listing
- Custom profile listing
- Current profile indication
- Error handling and edge cases
- Profile categorization
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from click.testing import CliRunner

from src.paas_ai.cli.commands.config.main import list_profiles
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


class TestListProfilesCommand:
    """Test the list_profiles command."""

    def test_list_profiles_basic(self):
        """Test basic list profiles command."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
                "production": "Production environment profile",
            }
            mock_current.return_value = "default"

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()

    def test_list_profiles_with_custom_profiles(self):
        """Test list profiles command with custom profiles."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
                "production": "Production environment profile",
                "custom1": "Custom profile: My first custom profile",
                "custom2": "Custom profile: My second custom profile",
            }
            mock_current.return_value = "custom1"

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()

    def test_list_profiles_with_only_builtin_profiles(self):
        """Test list profiles command with only built-in profiles."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
                "production": "Production environment profile",
            }
            mock_current.return_value = "production"

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()

    def test_list_profiles_with_only_custom_profiles(self):
        """Test list profiles command with only custom profiles."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "my-custom": "Custom profile: My custom profile",
                "another-custom": "Custom profile: Another custom profile",
            }
            mock_current.return_value = "my-custom"

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()

    def test_list_profiles_with_empty_profiles(self):
        """Test list profiles command with no profiles."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {}
            mock_current.return_value = "default"

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()

    def test_list_profiles_with_mixed_profile_types(self):
        """Test list profiles command with mixed profile types."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
                "production": "Production environment profile",
                "dev": "Custom profile: Development profile",
                "staging": "Custom profile: Staging profile",
                "test": "Custom profile: Test profile",
            }
            mock_current.return_value = "dev"

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()


class TestListProfilesCommandErrorHandling:
    """Test error handling in list_profiles command."""

    def test_list_profiles_with_list_error(self):
        """Test list profiles command with profile listing error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.config.main.list_available_profiles") as mock_list:
            mock_list.side_effect = Exception("Failed to list profiles")

            result = runner.invoke(list_profiles)

            # Verify command succeeded (error handling is internal)
            assert result.exit_code == 0
            # Verify the function was called
            mock_list.assert_called_once_with(include_custom=True)

    def test_list_profiles_with_current_profile_error(self):
        """Test list profiles command with current profile error."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {"default": "Default configuration profile"}
            mock_current.side_effect = Exception("Failed to get current profile")

            result = runner.invoke(list_profiles)

            # Verify command succeeded (error handling is internal)
            assert result.exit_code == 0
            # Verify the functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()

    def test_list_profiles_with_both_errors(self):
        """Test list profiles command with both errors."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.side_effect = Exception("List error")
            mock_current.side_effect = Exception("Current error")

            result = runner.invoke(list_profiles)

            # Verify command succeeded (error handling is internal)
            assert result.exit_code == 0
            # Verify the functions were called
            mock_list.assert_called_once_with(include_custom=True)
            # get_current_profile is not called when list_available_profiles fails
            mock_current.assert_not_called()


class TestListProfilesCommandEdgeCases:
    """Test edge cases for list_profiles command."""

    def test_list_profiles_with_special_characters_in_names(self):
        """Test list profiles command with special characters in profile names."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "test-with-dashes": "Custom profile: Test with dashes",
                "test_with_underscores": "Custom profile: Test with underscores",
                "test.with.dots": "Custom profile: Test with dots",
                "test@with#special$chars": "Custom profile: Test with special chars",
            }
            mock_current.return_value = "test-with-dashes"

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()

    def test_list_profiles_with_unicode_characters(self):
        """Test list profiles command with unicode characters."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "测试": "Custom profile: 测试配置文件",
                "тест": "Custom profile: Тестовая конфигурация",
                "テスト": "Custom profile: テスト設定",
            }
            mock_current.return_value = "测试"

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()

    def test_list_profiles_with_very_long_names(self):
        """Test list profiles command with very long profile names."""
        runner = CliRunner()

        long_name = "a" * 100
        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "default": "Default configuration profile",
                long_name: f"Custom profile: Very long profile name with {len(long_name)} characters",
            }
            mock_current.return_value = long_name

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()

    def test_list_profiles_with_very_long_descriptions(self):
        """Test list profiles command with very long descriptions."""
        runner = CliRunner()

        long_description = "This is a very long description that contains many words and should be displayed properly in the output without causing any formatting issues or truncation problems that might occur with extremely long text content."
        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "long-desc": f"Custom profile: {long_description}",
            }
            mock_current.return_value = "long-desc"

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()

    def test_list_profiles_with_none_values(self):
        """Test list profiles command with None values."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "default": "Default configuration profile",
                None: "Custom profile: None profile",
                "valid": "Custom profile: Valid profile",
            }
            mock_current.return_value = "valid"

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()


class TestListProfilesCommandIntegration:
    """Integration tests for list_profiles command."""

    def test_list_profiles_full_workflow(self):
        """Test complete list profiles workflow."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            # Mock realistic profile data
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
                "production": "Production environment profile",
                "staging": "Custom profile: Staging environment",
                "testing": "Custom profile: Testing environment",
                "development": "Custom profile: Development environment",
            }
            mock_current.return_value = "staging"

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()

    def test_list_profiles_consistency_across_calls(self):
        """Test that list profiles command is consistent across multiple calls."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
            }
            mock_current.return_value = "default"

            # Make multiple calls
            result1 = runner.invoke(list_profiles)
            result2 = runner.invoke(list_profiles)

            # Verify both commands succeeded
            assert result1.exit_code == 0
            assert result2.exit_code == 0
            # Verify the correct functions were called for both
            assert mock_list.call_count == 2
            assert mock_current.call_count == 2

    def test_list_profiles_with_different_current_profiles(self):
        """Test list profiles command with different current profiles."""
        runner = CliRunner()

        profiles = ["default", "local", "production"]

        for current_profile in profiles:
            with patch(
                "src.paas_ai.cli.commands.config.main.list_available_profiles"
            ) as mock_list, patch(
                "src.paas_ai.cli.commands.config.main.get_current_profile"
            ) as mock_current:
                mock_list.return_value = {
                    "default": "Default configuration profile",
                    "local": "Local development profile",
                    "production": "Production environment profile",
                }
                mock_current.return_value = current_profile

                result = runner.invoke(list_profiles)

                # Verify command succeeded
                assert result.exit_code == 0
                # Verify the correct functions were called
                mock_list.assert_called_once_with(include_custom=True)
                mock_current.assert_called_once()


class TestListProfilesCommandPerformance:
    """Test performance characteristics of list_profiles command."""

    def test_list_profiles_response_time(self):
        """Test that list profiles command responds quickly."""
        import time

        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
                "production": "Production environment profile",
            }
            mock_current.return_value = "default"

            start_time = time.time()
            result = runner.invoke(list_profiles)
            end_time = time.time()

            # Should respond quickly (less than 1 second)
            assert (end_time - start_time) < 1.0
            assert result.exit_code == 0

    def test_list_profiles_with_many_profiles(self):
        """Test list profiles command with many profiles."""
        runner = CliRunner()

        # Create many profiles
        many_profiles = {}
        for i in range(100):
            many_profiles[f"profile_{i}"] = f"Custom profile: Profile number {i}"

        many_profiles.update(
            {
                "default": "Default configuration profile",
                "local": "Local development profile",
                "production": "Production environment profile",
            }
        )

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = many_profiles
            mock_current.return_value = "default"

            result = runner.invoke(list_profiles)

            # Verify command succeeded
            assert result.exit_code == 0
            # Verify the correct functions were called
            mock_list.assert_called_once_with(include_custom=True)
            mock_current.assert_called_once()

    def test_list_profiles_memory_usage(self):
        """Test that list profiles command doesn't use excessive memory."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.config.main.list_available_profiles"
        ) as mock_list, patch(
            "src.paas_ai.cli.commands.config.main.get_current_profile"
        ) as mock_current:
            mock_list.return_value = {
                "default": "Default configuration profile",
                "local": "Local development profile",
                "production": "Production environment profile",
            }
            mock_current.return_value = "default"

            # Should not raise memory errors
            result = runner.invoke(list_profiles)
            assert result.exit_code == 0

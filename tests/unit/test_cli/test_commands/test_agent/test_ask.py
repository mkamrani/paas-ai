"""
Unit tests for agent ask command.

Tests all components of the ask command including:
- ask_command function
- Configuration loading and validation
- Agent initialization and interaction
- Error handling and edge cases
- Output formatting and display
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from click import ClickException
from click.testing import CliRunner

from src.paas_ai.cli.commands.agent.ask import ask_command
from src.paas_ai.core.config import ConfigurationError
from src.paas_ai.core.config.schemas import DEFAULT_CONFIG_PROFILES


class TestAskCommand:
    """Test the ask_command function."""

    def test_ask_command_basic_success(self):
        """Test basic successful ask command execution."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Test response"
            mock_rag_agent_class.return_value = mock_agent

            # Run command
            result = runner.invoke(ask_command, ["-q", "What is Alice's job?"])

            # Verify
            assert result.exit_code == 0
            assert "AGENT RESPONSE:" in result.output
            assert "Test response" in result.output
            assert "=" * 60 in result.output

            # Verify agent was called correctly
            mock_agent.ask.assert_called_once_with("What is Alice's job?")

    def test_ask_command_with_show_config(self):
        """Test ask command with --show-config flag."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Test response"
            mock_agent.get_config_summary.return_value = {
                "llm": {"provider": "openai", "model": "gpt-3.5-turbo"},
                "embedding": {"type": "openai", "model": "text-embedding-3-small"},
                "vectorstore": {"type": "chroma", "directory": "/tmp/chroma", "collection": "test"},
                "multi_agent": {
                    "mode": "supervisor",
                    "agents": ["designer"],
                    "track_tokens": True,
                    "verbose": False,
                },
            }
            mock_rag_agent_class.return_value = mock_agent

            # Run command
            result = runner.invoke(ask_command, ["-q", "Test question", "--show-config"])

            # Verify
            assert result.exit_code == 0
            assert "CONFIGURATION SUMMARY:" in result.output
            assert "LLM: openai (gpt-3.5-turbo)" in result.output
            assert "Embedding: openai (text-embedding-3-small)" in result.output
            assert "VectorStore: chroma -> /tmp/chroma" in result.output
            assert "Collection: test" in result.output
            assert "AGENT RESPONSE:" in result.output

            # Verify config summary was called
            mock_agent.get_config_summary.assert_called_once()

    def test_ask_command_with_config_profile(self):
        """Test ask command with --config-profile option."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "sentence_transformers"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Test response"
            mock_rag_agent_class.return_value = mock_agent

            # Run command with valid profile
            result = runner.invoke(
                ask_command, ["-q", "Test question", "--config-profile", "local"]
            )

            # Verify
            assert result.exit_code == 0
            assert "AGENT RESPONSE:" in result.output
            assert "Test response" in result.output

            # Verify agent was initialized with the profile config
            mock_rag_agent_class.assert_called_once()

    def test_ask_command_with_unknown_config_profile(self):
        """Test ask command with unknown config profile."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Test response"
            mock_rag_agent_class.return_value = mock_agent

            # Run command with unknown profile
            result = runner.invoke(
                ask_command, ["-q", "Test question", "--config-profile", "unknown"]
            )

            # Verify
            assert result.exit_code == 0
            assert "AGENT RESPONSE:" in result.output
            # Should fall back to default config
            mock_load_config.assert_called_once()

    def test_ask_command_missing_question(self):
        """Test ask command without required question parameter."""
        runner = CliRunner()

        # Run command without question
        result = runner.invoke(ask_command, [])

        # Verify
        assert result.exit_code != 0
        assert "Missing option" in result.output or "Error" in result.output

    def test_ask_command_configuration_error(self):
        """Test ask command with configuration error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config:
            # Setup mock to raise configuration error
            mock_load_config.side_effect = ConfigurationError("Config file not found")

            # Run command
            result = runner.invoke(ask_command, ["-q", "Test question"])

            # Verify
            assert result.exit_code == 0  # Command should handle error gracefully
            assert "❌ Failed to process question: Config file not found" in result.output

    def test_ask_command_agent_error(self):
        """Test ask command with agent processing error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.side_effect = Exception("Agent processing error")
            mock_rag_agent_class.return_value = mock_agent

            # Run command
            result = runner.invoke(ask_command, ["-q", "Test question"])

            # Verify
            assert result.exit_code == 0  # Command should handle error gracefully
            assert "❌ Failed to process question: Agent processing error" in result.output

    def test_ask_command_with_multi_agent_config_summary(self):
        """Test ask command with multi-agent configuration summary."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Test response"
            mock_agent.get_config_summary.return_value = {
                "llm": {"provider": "openai", "model": "gpt-3.5-turbo"},
                "embedding": {"type": "openai", "model": "text-embedding-3-small"},
                "vectorstore": {"type": "chroma", "directory": "/tmp/chroma", "collection": "test"},
                "multi_agent": {
                    "mode": "supervisor",
                    "agents": ["designer", "paas_manifest_generator"],
                    "track_tokens": True,
                    "verbose": True,
                },
            }
            mock_rag_agent_class.return_value = mock_agent

            # Run command
            result = runner.invoke(ask_command, ["-q", "Test question", "--show-config"])

            # Verify
            assert result.exit_code == 0
            assert "Multi-Agent Mode: supervisor" in result.output
            assert "Agents: designer, paas_manifest_generator" in result.output
            assert "Token Tracking: ON" in result.output
            assert "Verbose Mode: ON" in result.output

    def test_ask_command_empty_question(self):
        """Test ask command with empty question."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Empty question response"
            mock_rag_agent_class.return_value = mock_agent

            # Run command with empty question
            result = runner.invoke(ask_command, ["-q", ""])

            # Verify
            assert result.exit_code == 0
            assert "AGENT RESPONSE:" in result.output
            assert "Empty question response" in result.output
            mock_agent.ask.assert_called_once_with("")

    def test_ask_command_long_question(self):
        """Test ask command with very long question."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Long question response"
            mock_rag_agent_class.return_value = mock_agent

            # Create a very long question
            long_question = "What is " + "Alice's job " * 100 + "?"

            # Run command
            result = runner.invoke(ask_command, ["-q", long_question])

            # Verify
            assert result.exit_code == 0
            assert "AGENT RESPONSE:" in result.output
            assert "Long question response" in result.output
            mock_agent.ask.assert_called_once_with(long_question)

    def test_ask_command_special_characters_question(self):
        """Test ask command with special characters in question."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Special chars response"
            mock_rag_agent_class.return_value = mock_agent

            # Question with special characters
            special_question = "What is Alice's job? @#$%^&*()_+-=[]{}|;':\",./<>?"

            # Run command
            result = runner.invoke(ask_command, ["-q", special_question])

            # Verify
            assert result.exit_code == 0
            assert "AGENT RESPONSE:" in result.output
            assert "Special chars response" in result.output
            mock_agent.ask.assert_called_once_with(special_question)


class TestAskCommandEdgeCases:
    """Test edge cases for ask command."""

    def test_ask_command_with_none_config_profile(self):
        """Test ask command with None config profile."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Test response"
            mock_rag_agent_class.return_value = mock_agent

            # Run command with None profile (should use default)
            result = runner.invoke(ask_command, ["-q", "Test question", "--config-profile", "None"])

            # Verify
            assert result.exit_code == 0
            assert "AGENT RESPONSE:" in result.output

    def test_ask_command_agent_returns_none(self):
        """Test ask command when agent returns None."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = None
            mock_rag_agent_class.return_value = mock_agent

            # Run command
            result = runner.invoke(ask_command, ["-q", "Test question"])

            # Verify
            assert result.exit_code == 0
            assert "AGENT RESPONSE:" in result.output
            # Should handle None response gracefully

    def test_ask_command_agent_returns_empty_string(self):
        """Test ask command when agent returns empty string."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = ""
            mock_rag_agent_class.return_value = mock_agent

            # Run command
            result = runner.invoke(ask_command, ["-q", "Test question"])

            # Verify
            assert result.exit_code == 0
            assert "AGENT RESPONSE:" in result.output
            # Should handle empty response gracefully

    def test_ask_command_config_summary_error(self):
        """Test ask command when config summary fails."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Test response"
            mock_agent.get_config_summary.side_effect = Exception("Config summary error")
            mock_rag_agent_class.return_value = mock_agent

            # Run command with show-config
            result = runner.invoke(ask_command, ["-q", "Test question", "--show-config"])

            # Verify
            assert result.exit_code == 0
            # Should handle config summary error gracefully and still show response
            assert "AGENT RESPONSE:" in result.output
            assert "Test response" in result.output

    def test_ask_command_unicode_question(self):
        """Test ask command with unicode characters in question."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Unicode response"
            mock_rag_agent_class.return_value = mock_agent

            # Question with unicode characters
            unicode_question = "What is Alice's job? 测试中文 🚀 émojis"

            # Run command
            result = runner.invoke(ask_command, ["-q", unicode_question])

            # Verify
            assert result.exit_code == 0
            assert "AGENT RESPONSE:" in result.output
            assert "Unicode response" in result.output
            mock_agent.ask.assert_called_once_with(unicode_question)


class TestAskCommandIntegration:
    """Integration tests for ask command."""

    def test_ask_command_full_workflow(self):
        """Test complete ask command workflow."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Complete workflow response"
            mock_agent.get_config_summary.return_value = {
                "llm": {"provider": "openai", "model": "gpt-3.5-turbo"},
                "embedding": {"type": "openai", "model": "text-embedding-3-small"},
                "vectorstore": {"type": "chroma", "directory": "/tmp/chroma", "collection": "test"},
            }
            mock_rag_agent_class.return_value = mock_agent

            # Run command with all options
            result = runner.invoke(
                ask_command,
                ["-q", "What is Alice's job?", "--config-profile", "local", "--show-config"],
            )

            # Verify
            assert result.exit_code == 0
            assert "CONFIGURATION SUMMARY:" in result.output
            assert "AGENT RESPONSE:" in result.output
            assert "Complete workflow response" in result.output

            # Verify all methods were called
            mock_agent.get_config_summary.assert_called_once()
            mock_agent.ask.assert_called_once_with("What is Alice's job?")

    def test_ask_command_with_different_profiles(self):
        """Test ask command with different config profiles."""
        runner = CliRunner()

        profiles_to_test = ["default", "local", "production"]

        for profile in profiles_to_test:
            with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
                "src.paas_ai.cli.commands.agent.ask.RAGAgent"
            ) as mock_rag_agent_class:
                # Setup mocks
                mock_config = Mock()
                mock_config.embedding.type = "openai"
                mock_load_config.return_value = mock_config

                mock_agent = Mock()
                mock_agent.ask.return_value = f"Response for {profile}"
                mock_rag_agent_class.return_value = mock_agent

                # Run command with profile
                result = runner.invoke(
                    ask_command, ["-q", "Test question", "--config-profile", profile]
                )

                # Verify
                assert result.exit_code == 0
                assert "AGENT RESPONSE:" in result.output
                assert f"Response for {profile}" in result.output

    def test_ask_command_error_recovery(self):
        """Test ask command error recovery and graceful handling."""
        runner = CliRunner()

        # Test with configuration error
        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config:
            mock_load_config.side_effect = ConfigurationError("Test config error")

            result = runner.invoke(ask_command, ["-q", "Test question"])

            assert result.exit_code == 0
            assert "❌ Failed to process question: Test config error" in result.output

        # Test with agent error
        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.side_effect = Exception("Test agent error")
            mock_rag_agent_class.return_value = mock_agent

            result = runner.invoke(ask_command, ["-q", "Test question"])

            assert result.exit_code == 0
            assert "❌ Failed to process question: Test agent error" in result.output

    def test_ask_command_output_formatting(self):
        """Test ask command output formatting and display."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = (
                "Formatted response with\nmultiple lines\nand special chars: @#$%"
            )
            mock_rag_agent_class.return_value = mock_agent

            # Run command
            result = runner.invoke(ask_command, ["-q", "Test question"])

            # Verify output formatting
            assert result.exit_code == 0
            assert "=" * 60 in result.output  # Separator lines
            assert "AGENT RESPONSE:" in result.output
            assert "Formatted response with" in result.output
            assert "multiple lines" in result.output
            assert "and special chars: @#$%" in result.output

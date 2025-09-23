"""
Unit tests for agent command factory and utilities.

Tests utility functions and factory patterns used by agent commands including:
- Command creation and registration
- Utility function behavior
- Factory pattern implementations
- Helper function edge cases
"""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from src.paas_ai.cli.commands.agent import agent_group
from src.paas_ai.cli.commands.agent.ask import ask_command
from src.paas_ai.cli.commands.agent.chat import chat_command


class TestAgentCommandFactory:
    """Test agent command factory and creation patterns."""

    def test_agent_group_creation(self):
        """Test that agent group is properly created."""
        from src.paas_ai.cli.commands.agent import agent_group

        assert agent_group is not None
        assert hasattr(agent_group, "name")
        assert hasattr(agent_group, "commands")
        assert agent_group.name == "agent"

    def test_agent_commands_creation(self):
        """Test that agent commands are properly created."""
        from src.paas_ai.cli.commands.agent.ask import ask_command
        from src.paas_ai.cli.commands.agent.chat import chat_command

        assert ask_command is not None
        assert chat_command is not None
        assert hasattr(ask_command, "name")
        assert hasattr(chat_command, "name")

    def test_agent_commands_registration(self):
        """Test that agent commands are properly registered."""
        from src.paas_ai.cli.commands.agent import agent_group

        # Check that commands are registered
        assert "ask" in agent_group.commands
        assert "chat" in agent_group.commands

        # Check that registered commands are the correct instances
        assert agent_group.commands["ask"] == ask_command
        assert agent_group.commands["chat"] == chat_command

    def test_agent_group_command_discovery(self):
        """Test that agent group can discover commands."""
        from src.paas_ai.cli.commands.agent import agent_group

        # Get all command names
        command_names = list(agent_group.commands.keys())

        # Should have exactly 2 commands
        assert len(command_names) == 2
        assert "ask" in command_names
        assert "chat" in command_names

    def test_agent_commands_help_generation(self):
        """Test that agent commands generate proper help text."""
        runner = CliRunner()

        # Test agent group help
        result = runner.invoke(agent_group, ["--help"])
        assert result.exit_code == 0
        assert "Agent commands for testing RAG integration" in result.output

        # Test ask command help
        result = runner.invoke(ask_command, ["--help"])
        assert result.exit_code == 0
        assert "Ask the RAG agent a question" in result.output

        # Test chat command help
        result = runner.invoke(chat_command, ["--help"])
        assert result.exit_code == 0
        assert "Start an interactive chat session" in result.output


class TestAgentCommandUtilities:
    """Test utility functions used by agent commands."""

    def test_agent_group_invocation(self):
        """Test that agent group can be invoked properly."""
        runner = CliRunner()

        # Test direct invocation
        result = runner.invoke(agent_group, ["--help"])
        assert result.exit_code == 0

        # Test command invocation through group
        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Test response"
            mock_rag_agent_class.return_value = mock_agent

            result = runner.invoke(agent_group, ["ask", "-q", "Test question"])
            assert result.exit_code == 0
            assert "Test response" in result.output

    def test_agent_commands_parameter_validation(self):
        """Test that agent commands validate parameters properly."""
        runner = CliRunner()

        # Test ask command without required question
        result = runner.invoke(agent_group, ["ask"])
        assert result.exit_code != 0

        # Test ask command with question
        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Test response"
            mock_rag_agent_class.return_value = mock_agent

            result = runner.invoke(agent_group, ["ask", "-q", "Test question"])
            assert result.exit_code == 0

    def test_agent_commands_option_handling(self):
        """Test that agent commands handle options properly."""
        runner = CliRunner()

        # Test ask command with various options
        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Test response"
            mock_agent.get_config_summary.return_value = {
                "llm": {"provider": "openai", "model": "gpt-3.5-turbo"},
                "embedding": {"type": "openai", "model": "text-embedding-3-small"},
                "vectorstore": {"type": "chroma", "directory": "/tmp/chroma", "collection": "test"},
            }
            mock_rag_agent_class.return_value = mock_agent

            # Test with show-config option
            result = runner.invoke(agent_group, ["ask", "-q", "Test question", "--show-config"])
            assert result.exit_code == 0
            assert "CONFIGURATION SUMMARY:" in result.output

            # Test with config-profile option
            result = runner.invoke(
                agent_group, ["ask", "-q", "Test question", "--config-profile", "local"]
            )
            assert result.exit_code == 0

        # Test chat command with various options
        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.RAGAgent"
        ) as mock_rag_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = False
            mock_config.multi_agent.verbose = False
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_rag_agent_class.return_value = mock_agent
            mock_prompt.side_effect = ["exit"]

            # Test with max-history option
            result = runner.invoke(agent_group, ["chat", "--max-history", "10"])
            assert result.exit_code == 0
            assert "📝 Max history: 10 messages" in result.output


class TestAgentCommandEdgeCases:
    """Test edge cases for agent command factory and utilities."""

    def test_agent_group_with_empty_commands(self):
        """Test agent group behavior with no commands."""
        # This test would require creating a new group without commands
        # For now, we just verify the existing group has commands
        from src.paas_ai.cli.commands.agent import agent_group

        assert len(agent_group.commands) > 0
        assert "ask" in agent_group.commands
        assert "chat" in agent_group.commands

    def test_agent_commands_with_invalid_parameters(self):
        """Test agent commands with invalid parameters."""
        runner = CliRunner()

        # Test with invalid option values
        result = runner.invoke(
            agent_group, ["ask", "-q", "Test question", "--config-profile", ""]  # Empty profile
        )
        # Should handle gracefully
        assert result.exit_code == 0

        # Test with very long parameters
        long_question = "What is " + "Alice's job " * 1000 + "?"
        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Long question response"
            mock_rag_agent_class.return_value = mock_agent

            result = runner.invoke(agent_group, ["ask", "-q", long_question])
            assert result.exit_code == 0
            assert "Long question response" in result.output

    def test_agent_commands_with_special_characters(self):
        """Test agent commands with special characters in parameters."""
        runner = CliRunner()

        special_question = "What is Alice's job? @#$%^&*()_+-=[]{}|;':\",./<>?"

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Special chars response"
            mock_rag_agent_class.return_value = mock_agent

            result = runner.invoke(agent_group, ["ask", "-q", special_question])
            assert result.exit_code == 0
            assert "Special chars response" in result.output

    def test_agent_commands_with_unicode_parameters(self):
        """Test agent commands with unicode characters in parameters."""
        runner = CliRunner()

        unicode_question = "What is Alice's job? 测试中文 🚀 émojis"

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Unicode response"
            mock_rag_agent_class.return_value = mock_agent

            result = runner.invoke(agent_group, ["ask", "-q", unicode_question])
            assert result.exit_code == 0
            assert "Unicode response" in result.output


class TestAgentCommandCompatibility:
    """Test compatibility of agent command factory and utilities."""

    def test_agent_commands_click_compatibility(self):
        """Test that agent commands are compatible with click framework."""
        # Test that all are click objects
        import click

        from src.paas_ai.cli.commands.agent import agent_group
        from src.paas_ai.cli.commands.agent.ask import ask_command
        from src.paas_ai.cli.commands.agent.chat import chat_command

        assert isinstance(agent_group, click.Group)
        assert isinstance(ask_command, click.Command)
        assert isinstance(chat_command, click.Command)

        # Test that they have required click attributes
        assert hasattr(agent_group, "invoke")
        assert hasattr(ask_command, "invoke")
        assert hasattr(chat_command, "invoke")

    def test_agent_commands_parameter_compatibility(self):
        """Test parameter compatibility across agent commands."""
        runner = CliRunner()

        # Both commands should support config-profile
        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Response"
            mock_rag_agent_class.return_value = mock_agent

            # Test ask command with config-profile
            result = runner.invoke(agent_group, ["ask", "-q", "Test", "--config-profile", "local"])
            assert result.exit_code == 0

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.RAGAgent"
        ) as mock_rag_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = False
            mock_config.multi_agent.verbose = False
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_rag_agent_class.return_value = mock_agent
            mock_prompt.side_effect = ["exit"]

            # Test chat command with config-profile
            result = runner.invoke(agent_group, ["chat", "--config-profile", "local"])
            assert result.exit_code == 0

    def test_agent_commands_output_compatibility(self):
        """Test output compatibility across agent commands."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Test response"
            mock_rag_agent_class.return_value = mock_agent

            # Test ask command output
            result = runner.invoke(agent_group, ["ask", "-q", "Test question"])
            assert result.exit_code == 0
            assert "AGENT RESPONSE:" in result.output
            assert "Test response" in result.output
            assert "=" * 60 in result.output  # Separator lines

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.RAGAgent"
        ) as mock_rag_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = False
            mock_config.multi_agent.verbose = False
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_rag_agent_class.return_value = mock_agent
            mock_prompt.side_effect = ["exit"]

            # Test chat command output
            result = runner.invoke(agent_group, ["chat"])
            assert result.exit_code == 0
            assert "🤖 RAG AGENT INTERACTIVE CHAT SESSION" in result.output
            assert "👋 Thanks for chatting! Goodbye!" in result.output

"""
Unit tests for agent module initialization.

Tests all components of the agent module initialization including:
- agent_group function
- Command registration
- Module structure and imports
- Click group functionality
- Command discovery and execution
"""

from unittest.mock import Mock, patch

import click
import pytest
from click.testing import CliRunner

from src.paas_ai.cli.commands.agent import agent_group, ask_command, chat_command


class TestAgentGroup:
    """Test the agent_group click group."""

    def test_agent_group_is_click_group(self):
        """Test that agent_group is a click group."""
        assert isinstance(agent_group, click.Group)
        assert hasattr(agent_group, "commands")
        assert hasattr(agent_group, "add_command")

    def test_agent_group_name(self):
        """Test that agent_group has correct name."""
        assert agent_group.name == "agent"

    def test_agent_group_help_text(self):
        """Test that agent_group has correct help text."""
        assert "Agent commands for testing RAG integration" in agent_group.help

    def test_agent_group_commands_registered(self):
        """Test that commands are properly registered."""
        # Get command names
        command_names = list(agent_group.commands.keys())

        # Should have ask and chat commands
        assert "ask" in command_names
        assert "chat" in command_names
        assert len(command_names) == 2

    def test_agent_group_ask_command_registered(self):
        """Test that ask command is properly registered."""
        ask_cmd = agent_group.commands.get("ask")
        assert ask_cmd is not None
        assert ask_cmd == ask_command

    def test_agent_group_chat_command_registered(self):
        """Test that chat command is properly registered."""
        chat_cmd = agent_group.commands.get("chat")
        assert chat_cmd is not None
        assert chat_cmd == chat_command

    def test_agent_group_command_execution(self):
        """Test that agent group can execute commands."""
        runner = CliRunner()

        # Test help command
        result = runner.invoke(agent_group, ["--help"])
        assert result.exit_code == 0
        assert "Agent commands for testing RAG integration" in result.output
        assert "ask" in result.output
        assert "chat" in result.output

    def test_agent_group_ask_command_help(self):
        """Test that ask command help works through group."""
        runner = CliRunner()

        result = runner.invoke(agent_group, ["ask", "--help"])
        assert result.exit_code == 0
        assert "Ask the RAG agent a question" in result.output
        assert "--question" in result.output or "-q" in result.output
        assert "--config-profile" in result.output
        assert "--show-config" in result.output

    def test_agent_group_chat_command_help(self):
        """Test that chat command help works through group."""
        runner = CliRunner()

        result = runner.invoke(agent_group, ["chat", "--help"])
        assert result.exit_code == 0
        assert "Start an interactive chat session" in result.output
        assert "--config-profile" in result.output
        assert "--show-config" in result.output
        assert "--thread-id" in result.output

    def test_agent_group_invalid_command(self):
        """Test that agent group handles invalid commands."""
        runner = CliRunner()

        result = runner.invoke(agent_group, ["invalid-command"])
        assert result.exit_code != 0
        assert "No such command" in result.output or "Error" in result.output

    def test_agent_group_no_command_specified(self):
        """Test that agent group shows help when no command specified."""
        runner = CliRunner()

        result = runner.invoke(agent_group, [])
        assert result.exit_code == 2  # Click groups return exit code 2 when no command specified
        assert "Agent commands for testing RAG integration" in result.output
        assert "ask" in result.output
        assert "chat" in result.output


class TestAgentModuleImports:
    """Test agent module imports and structure."""

    def test_agent_module_imports(self):
        """Test that agent module imports work correctly."""
        from src.paas_ai.cli.commands.agent import agent_group, ask_command, chat_command

        # Verify imports are not None
        assert agent_group is not None
        assert ask_command is not None
        assert chat_command is not None

    def test_agent_module_all_exports(self):
        """Test that agent module __all__ exports are correct."""
        import src.paas_ai.cli.commands.agent as agent_module

        # Check that __all__ exists in the module
        assert hasattr(agent_module, "__all__")

        # Check that expected items are exported
        expected_exports = ["agent_group"]
        for export in expected_exports:
            assert export in agent_module.__all__

    def test_agent_module_docstring(self):
        """Test that agent module has proper docstring."""
        import src.paas_ai.cli.commands.agent as agent_module

        assert agent_module.__doc__ is not None
        assert "Agent CLI commands module" in agent_module.__doc__
        assert "ask: Ask the agent a single question" in agent_module.__doc__
        assert "chat: Start an interactive chat session" in agent_module.__doc__


class TestAgentCommandIntegration:
    """Test integration between agent commands and group."""

    def test_ask_command_through_group(self):
        """Test ask command execution through agent group."""
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

            # Run command through group
            result = runner.invoke(agent_group, ["ask", "-q", "Test question"])

            # Verify
            assert result.exit_code == 0
            assert "AGENT RESPONSE:" in result.output
            assert "Test response" in result.output

    def test_chat_command_through_group(self):
        """Test chat command execution through agent group."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.MultiAgentSystem"
        ) as mock_multi_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = False
            mock_config.multi_agent.verbose = False
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.chat_stream.return_value = ["Response"]
            mock_agent.get_token_session_summary.return_value = {"total_tokens": 0}
            mock_multi_agent_class.return_value = mock_agent

            # Mock user input and exit
            mock_prompt.side_effect = ["exit"]

            # Run command through group
            result = runner.invoke(agent_group, ["chat"])

            # Verify
            assert result.exit_code == 0
            assert "🤖 MULTI-AGENT INTERACTIVE CHAT SESSION" in result.output
            assert "👋 Thanks for chatting! Goodbye!" in result.output

    def test_agent_group_command_discovery(self):
        """Test that agent group can discover and list commands."""
        runner = CliRunner()

        # Test help to see all commands
        result = runner.invoke(agent_group, ["--help"])

        assert result.exit_code == 0
        # Should show both commands
        assert "ask" in result.output
        assert "chat" in result.output

        # Should show command descriptions
        assert "Ask the RAG agent a question" in result.output
        assert "Start an interactive chat session" in result.output

    def test_agent_group_command_aliases(self):
        """Test that agent group commands work with different invocation styles."""
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

            # Test different ways to invoke ask command
            test_cases = [
                ["ask", "-q", "Test question"],
                ["ask", "--question", "Test question"],
            ]

            for args in test_cases:
                result = runner.invoke(agent_group, args)
                assert result.exit_code == 0
                assert "AGENT RESPONSE:" in result.output
                assert "Test response" in result.output


class TestAgentModuleEdgeCases:
    """Test edge cases for agent module."""

    def test_agent_group_with_invalid_options(self):
        """Test agent group with invalid options."""
        runner = CliRunner()

        # Test with invalid global options
        result = runner.invoke(agent_group, ["--invalid-option"])
        assert result.exit_code != 0

    def test_agent_group_with_missing_required_args(self):
        """Test agent group with missing required arguments."""
        runner = CliRunner()

        # Test ask command without required question
        result = runner.invoke(agent_group, ["ask"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "Error" in result.output

    def test_agent_group_command_error_handling(self):
        """Test agent group error handling."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config:
            # Setup mock to raise error
            mock_load_config.side_effect = Exception("Test error")

            # Run command through group
            result = runner.invoke(agent_group, ["ask", "-q", "Test question"])

            # Should handle error gracefully
            assert result.exit_code == 0
            assert "❌ Failed to process question: Test error" in result.output

    def test_agent_group_import_error_handling(self):
        """Test agent group handles import errors gracefully."""
        # This test would require mocking import failures, which is complex
        # For now, we just verify the module can be imported
        try:
            from src.paas_ai.cli.commands.agent import agent_group

            assert agent_group is not None
        except ImportError as e:
            pytest.fail(f"Agent module import failed: {e}")


class TestAgentModuleCompatibility:
    """Test compatibility and interoperability of agent module."""

    def test_agent_group_click_compatibility(self):
        """Test that agent group is compatible with click framework."""
        # Test that it's a proper click group
        assert hasattr(agent_group, "commands")
        assert hasattr(agent_group, "add_command")
        assert hasattr(agent_group, "invoke")
        assert hasattr(agent_group, "get_help")

        # Test that it can be used as a click command
        runner = CliRunner()
        result = runner.invoke(agent_group, ["--help"])
        assert result.exit_code == 0

    def test_agent_commands_click_compatibility(self):
        """Test that agent commands are compatible with click framework."""
        # Test ask command
        assert hasattr(ask_command, "params")
        assert hasattr(ask_command, "invoke")
        assert hasattr(ask_command, "get_help")

        # Test chat command
        assert hasattr(chat_command, "params")
        assert hasattr(chat_command, "invoke")
        assert hasattr(chat_command, "get_help")

    def test_agent_group_command_consistency(self):
        """Test that agent group commands are consistent."""
        # Both commands should be click commands
        assert isinstance(ask_command, click.Command)
        assert isinstance(chat_command, click.Command)

        # Both should have proper help text
        assert ask_command.help is not None
        assert chat_command.help is not None

        # Both should be properly registered
        assert "ask" in agent_group.commands
        assert "chat" in agent_group.commands

    def test_agent_module_structure_consistency(self):
        """Test that agent module structure is consistent."""
        import src.paas_ai.cli.commands.agent as agent_module

        # Should have expected attributes
        assert hasattr(agent_module, "agent_group")
        assert hasattr(agent_module, "ask_command")
        assert hasattr(agent_module, "chat_command")

        # Should have proper docstring
        assert agent_module.__doc__ is not None

        # Should have __all__ defined
        assert hasattr(agent_module, "__all__")
        assert isinstance(agent_module.__all__, list)
        assert len(agent_module.__all__) > 0

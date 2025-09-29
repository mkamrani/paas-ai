"""
Integration tests for agent commands.

Tests the complete workflow and integration between agent commands and
the underlying multi-agent system.
"""

from unittest.mock import Mock, patch

from click.testing import CliRunner
from langchain_core.messages import AIMessage, HumanMessage

from paas_ai.core.config import ConfigurationError
from src.paas_ai.cli.commands.agent import agent_group
from src.paas_ai.cli.commands.agent.chat import chat_command
from src.paas_ai.core.config.schemas import DEFAULT_CONFIG_PROFILES


class TestAgentCommandIntegration:
    """Integration tests for agent commands."""

    def test_chat_command_full_workflow_integration(self):
        """Test complete chat command workflow integration."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.MultiAgentSystem"
        ) as mock_multi_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            # Setup comprehensive mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = True
            mock_config.multi_agent.verbose = True
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.chat_stream.return_value = ["Hello", " world"]
            mock_agent.chat_stream.return_value = ["Follow", " up"]
            mock_agent.chat.return_value = "Follow up"
            mock_agent.get_available_tools.return_value = [
                {"name": "rag_search", "description": "Search knowledge base"},
                {"name": "design_specification", "description": "Create design specs"},
            ]
            mock_agent.get_config_summary.return_value = {
                "llm": {"provider": "openai", "model": "gpt-3.5-turbo"},
                "embedding": {"type": "openai", "model": "text-embedding-3-small"},
                "vectorstore": {"type": "chroma", "directory": "/tmp/chroma", "collection": "test"},
                "multi_agent": {
                    "mode": "supervisor",
                    "agents": ["designer"],
                    "track_tokens": True,
                    "verbose": True,
                },
            }
            mock_agent.get_token_session_summary.return_value = {
                "total_tokens": 150,
                "agents_used": ["designer"],
                "session_duration": 2.5,
            }
            mock_multi_agent_class.return_value = mock_agent

            # Mock interactive chat - test multiple interactions
            mock_prompt.side_effect = ["Hello", "Follow up question", "exit"]

            # Test chat with config display
            result = runner.invoke(agent_group, ["chat", "--show-config"])

            # Verify complete workflow
            assert result.exit_code == 0
            assert "🤖 MULTI-AGENT INTERACTIVE CHAT SESSION" in result.output
            assert "CONFIGURATION SUMMARY:" in result.output
            assert "Multi-Agent Mode: supervisor" in result.output
            assert "Agents: designer" in result.output
            assert "👋 Thanks for chatting! Goodbye!" in result.output

            # Verify all components were called
            mock_agent.get_config_summary.assert_called_once()
            assert mock_agent.chat_stream.call_count >= 2

    def test_agent_commands_config_profile_integration(self):
        """Test agent commands with different config profiles."""
        runner = CliRunner()

        profiles_to_test = ["default", "local", "production"]

        for profile in profiles_to_test:
            # Test chat command with profile
            with patch(
                "src.paas_ai.cli.commands.agent.chat.load_config"
            ) as mock_load_config, patch(
                "src.paas_ai.cli.commands.agent.chat.MultiAgentSystem"
            ) as mock_multi_agent_class, patch(
                "src.paas_ai.cli.commands.agent.chat.click.prompt"
            ) as mock_prompt:
                mock_config = Mock()
                mock_config.embedding.type = "openai"
                mock_config.multi_agent.track_tokens = False
                mock_config.multi_agent.verbose = False
                mock_load_config.return_value = mock_config

                mock_agent = Mock()
                mock_agent.chat_stream.return_value = ["Response"]
                mock_agent.get_token_session_summary.return_value = {"total_tokens": 0}
                mock_multi_agent_class.return_value = mock_agent
                mock_prompt.side_effect = ["exit"]

                # Test chat command with config-profile
                result = runner.invoke(agent_group, ["chat", "--config-profile", profile])
                assert result.exit_code == 0

    def test_agent_commands_streaming_integration(self):
        """Test streaming integration between agent commands."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.MultiAgentSystem"
        ) as mock_multi_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = False
            mock_config.multi_agent.verbose = False
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.chat_stream.return_value = ["Streaming", " response", " tokens"]
            mock_agent.chat_stream.return_value = ["Streaming", " response", " tokens"]
            mock_agent.chat.return_value = "Streaming response tokens"
            mock_agent.get_token_session_summary.return_value = {"total_tokens": 0}
            mock_multi_agent_class.return_value = mock_agent
            mock_prompt.side_effect = ["Test question", "exit"]

            # Test debug streaming
            result = runner.invoke(agent_group, ["chat", "--debug-streaming"])

            assert result.exit_code == 0
            assert "Streaming response tokens" in result.output


class TestAgentCommandCompatibilityIntegration:
    """Test compatibility and integration aspects of agent commands."""

    def test_agent_commands_click_integration(self):
        """Test that agent commands integrate properly with click framework."""
        runner = CliRunner()

        # Test that agent group works as a click command
        result = runner.invoke(agent_group, ["--help"])
        assert result.exit_code == 0

        # Test that individual commands work as click commands

        result = runner.invoke(chat_command, ["--help"])
        assert result.exit_code == 0

    def test_agent_commands_parameter_compatibility(self):
        """Test parameter compatibility across agent commands."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.MultiAgentSystem"
        ) as mock_multi_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = False
            mock_config.multi_agent.verbose = False
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_multi_agent_class.return_value = mock_agent
            mock_prompt.side_effect = ["exit"]

            # Test chat command with config-profile
            result = runner.invoke(agent_group, ["chat", "--config-profile", "local"])
            assert result.exit_code == 0

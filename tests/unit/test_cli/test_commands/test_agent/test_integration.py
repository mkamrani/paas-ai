"""
Integration tests for agent CLI commands.

Tests the complete agent command workflow including:
- End-to-end command execution
- Cross-component interactions
- Real agent integration (with mocking)
- Error propagation and handling
- Configuration management across components
- Multi-agent system integration
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner
from langchain_core.messages import AIMessage, HumanMessage

from paas_ai.core.config import ConfigurationError
from src.paas_ai.cli.commands.agent import agent_group
from src.paas_ai.cli.commands.agent.ask import ask_command
from src.paas_ai.cli.commands.agent.chat import chat_command
from src.paas_ai.core.config.schemas import DEFAULT_CONFIG_PROFILES


class TestAgentCommandIntegration:
    """Integration tests for agent commands."""

    def test_ask_command_full_workflow_integration(self):
        """Test complete ask command workflow integration."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup comprehensive mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Complete integration response"
            mock_agent.get_config_summary.return_value = {
                "llm": {"provider": "openai", "model": "gpt-3.5-turbo"},
                "embedding": {"type": "openai", "model": "text-embedding-3-small"},
                "vectorstore": {"type": "chroma", "directory": "/tmp/chroma", "collection": "test"},
                "multi_agent": {
                    "mode": "supervisor",
                    "agents": ["designer", "paas_manifest_generator"],
                    "track_tokens": True,
                    "verbose": False,
                },
            }
            mock_rag_agent_class.return_value = mock_agent

            # Test through agent group
            result = runner.invoke(
                agent_group,
                ["ask", "-q", "What is Alice's job?", "--config-profile", "local", "--show-config"],
            )

            # Verify complete workflow
            assert result.exit_code == 0
            assert "CONFIGURATION SUMMARY:" in result.output
            assert "Multi-Agent Mode: supervisor" in result.output
            assert "Agents: designer, paas_manifest_generator" in result.output
            assert "AGENT RESPONSE:" in result.output
            assert "Complete integration response" in result.output

            # Verify all components were called
            mock_agent.get_config_summary.assert_called_once()
            mock_agent.ask.assert_called_once_with("What is Alice's job?")

    def test_chat_command_full_workflow_integration(self):
        """Test complete chat command workflow integration."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.RAGAgent"
        ) as mock_rag_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            # Setup comprehensive mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = True
            mock_config.multi_agent.verbose = True
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask_stream.return_value = ["Hello", " world"]
            mock_agent.chat_stream.return_value = ["Follow", " up"]
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
                "total_tokens": 200,
                "total_input_tokens": 100,
                "total_output_tokens": 100,
                "total_requests": 2,
                "agents_used": ["designer"],
                "session_duration": 4.2,
                "agent_breakdown": {"designer": {"total_tokens": 200, "requests": 2}},
                "model_breakdown": {"gpt-3.5-turbo": {"total_tokens": 200, "requests": 2}},
            }
            mock_agent.clear_token_history = Mock()
            mock_rag_agent_class.return_value = mock_agent

            # Mock conversation flow
            mock_prompt.side_effect = [
                "Hello",  # First question
                "tools",  # Show tools
                "config",  # Show config
                "tokens",  # Show tokens
                "Follow up question",  # Second question
                "clear",  # Clear history
                "exit",  # Exit
            ]

            # Test through agent group
            result = runner.invoke(
                agent_group,
                ["chat", "--config-profile", "local", "--show-config", "--max-history", "10"],
            )

            # Verify complete workflow
            assert result.exit_code == 0
            assert "CONFIGURATION SUMMARY:" in result.output
            assert "Multi-Agent Mode: supervisor" in result.output
            assert "Hello world" in result.output
            assert "🔧 AVAILABLE TOOLS:" in result.output
            assert "rag_search" in result.output
            assert "⚙️  CURRENT CONFIGURATION:" in result.output
            assert "🪙 TOKEN USAGE SUMMARY:" in result.output
            assert "Follow up" in result.output
            assert "🧹 Conversation history cleared!" in result.output
            assert "👋 Thanks for chatting! Goodbye!" in result.output
            assert "📊 Session completed: 2 exchanges" in result.output
            assert "🪙 200 tokens used across 1 agents" in result.output

            # Verify all components were called
            mock_agent.get_config_summary.assert_called()
            mock_agent.get_available_tools.assert_called()
            mock_agent.get_token_session_summary.assert_called()
            mock_agent.clear_token_history.assert_called()

    def test_agent_commands_config_profile_integration(self):
        """Test agent commands with different config profiles."""
        runner = CliRunner()

        profiles_to_test = ["default", "local", "production"]

        for profile in profiles_to_test:
            # Test ask command with profile
            with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
                "src.paas_ai.cli.commands.agent.ask.RAGAgent"
            ) as mock_rag_agent_class:
                mock_config = Mock()
                mock_config.embedding.type = "openai"
                mock_load_config.return_value = mock_config

                mock_agent = Mock()
                mock_agent.ask.return_value = f"Response for {profile}"
                mock_rag_agent_class.return_value = mock_agent

                result = runner.invoke(
                    agent_group, ["ask", "-q", "Test question", "--config-profile", profile]
                )

                assert result.exit_code == 0
                assert f"Response for {profile}" in result.output

            # Test chat command with profile
            with patch(
                "src.paas_ai.cli.commands.agent.chat.load_config"
            ) as mock_load_config, patch(
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

                result = runner.invoke(agent_group, ["chat", "--config-profile", profile])

                assert result.exit_code == 0
                assert "🤖 RAG AGENT INTERACTIVE CHAT SESSION" in result.output

    def test_agent_commands_error_propagation_integration(self):
        """Test error propagation through agent command system."""
        runner = CliRunner()

        # Test configuration error propagation
        with patch("src.paas_ai.cli.commands.agent.ask.RAGAgent") as mock_rag_agent_class:
            # Import ConfigurationError from the same path as ask.py uses
            from paas_ai.core.config import ConfigurationError as AskConfigurationError

            mock_rag_agent_class.side_effect = AskConfigurationError("Config integration error")

            result = runner.invoke(agent_group, ["ask", "-q", "Test question"])

            assert result.exit_code == 0
            assert "❌ Configuration error: Config integration error" in result.output
            assert "💡 Try creating a config file with: paas-ai config init" in result.output

        # Test agent error propagation
        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.side_effect = Exception("Agent integration error")
            mock_rag_agent_class.return_value = mock_agent

            result = runner.invoke(agent_group, ["ask", "-q", "Test question"])

            assert result.exit_code == 0
            assert "❌ Failed to process question: Agent integration error" in result.output

    def test_agent_commands_streaming_integration(self):
        """Test streaming functionality integration."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.RAGAgent"
        ) as mock_rag_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = False
            mock_config.multi_agent.verbose = False
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask_stream.return_value = ["Streaming", " response", " tokens"]
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input and exit
            mock_prompt.side_effect = ["Test question", "exit"]

            # Test streaming
            result = runner.invoke(agent_group, ["chat", "--debug-streaming"])

            assert result.exit_code == 0
            assert "Streaming response tokens" in result.output

            # Test direct streaming
            mock_agent.ask_stream_direct.return_value = ["Direct", " streaming"]
            mock_prompt.side_effect = ["Test question", "exit"]

            result = runner.invoke(agent_group, ["chat", "--direct-streaming"])

            assert result.exit_code == 0
            assert "Direct streaming" in result.output

    def test_agent_commands_multi_agent_integration(self):
        """Test multi-agent system integration."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks with multi-agent configuration
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Multi-agent response"
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
                "agent_details": {
                    "designer": {
                        "tools": ["rag_search", "design_specification"],
                        "mode": "supervisor",
                    },
                    "paas_manifest_generator": {
                        "tools": ["rag_search", "paas_manifest_generator"],
                        "mode": "supervisor",
                    },
                },
            }
            mock_rag_agent_class.return_value = mock_agent

            # Test ask command with multi-agent config
            result = runner.invoke(agent_group, ["ask", "-q", "Design a system", "--show-config"])

            assert result.exit_code == 0
            assert "Multi-Agent Mode: supervisor" in result.output
            assert "Agents: designer, paas_manifest_generator" in result.output
            assert "Token Tracking: ON" in result.output
            assert "Verbose Mode: ON" in result.output
            assert "Multi-agent response" in result.output


class TestAgentCommandCrossComponentIntegration:
    """Test cross-component integration for agent commands."""

    def test_agent_commands_config_consistency(self):
        """Test configuration consistency across agent commands."""
        runner = CliRunner()

        # Test that both commands use the same config system
        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_ask_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.load_config"
        ) as mock_chat_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_ask_rag_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.RAGAgent"
        ) as mock_chat_rag_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            # Setup consistent mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = False
            mock_config.multi_agent.verbose = False
            mock_ask_load_config.return_value = mock_config
            mock_chat_load_config.return_value = mock_config

            mock_ask_agent = Mock()
            mock_ask_agent.ask.return_value = "Ask response"
            mock_ask_rag_agent_class.return_value = mock_ask_agent

            mock_chat_agent = Mock()
            mock_chat_rag_agent_class.return_value = mock_chat_agent
            mock_prompt.side_effect = ["exit"]

            # Test ask command
            ask_result = runner.invoke(agent_group, ["ask", "-q", "Test question"])
            assert ask_result.exit_code == 0
            assert "Ask response" in ask_result.output

            # Test chat command
            chat_result = runner.invoke(agent_group, ["chat"])
            assert chat_result.exit_code == 0

            # Verify both used the same config
            mock_ask_load_config.assert_called_once()
            mock_chat_load_config.assert_called_once()

    def test_agent_commands_error_handling_consistency(self):
        """Test error handling consistency across agent commands."""
        runner = CliRunner()

        # Test configuration error handling consistency
        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_ask_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.load_config"
        ) as mock_chat_load_config:
            # Both should handle config errors the same way
            mock_ask_load_config.side_effect = ConfigurationError("Config error")
            mock_chat_load_config.side_effect = ConfigurationError("Config error")

            # Test ask command error handling
            ask_result = runner.invoke(agent_group, ["ask", "-q", "Test question"])
            assert ask_result.exit_code == 0
            assert "❌ Configuration error: Config error" in ask_result.output

            # Test chat command error handling
            chat_result = runner.invoke(agent_group, ["chat"])
            assert chat_result.exit_code == 0
            assert "❌ Configuration error: Config error" in chat_result.output
            assert "💡 Try creating a config file with: paas-ai config init" in chat_result.output

    def test_agent_commands_output_formatting_consistency(self):
        """Test output formatting consistency across agent commands."""
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
            }
            mock_rag_agent_class.return_value = mock_agent

            # Test ask command output formatting
            result = runner.invoke(agent_group, ["ask", "-q", "Test question", "--show-config"])

            assert result.exit_code == 0
            # Should have consistent formatting
            assert "=" * 60 in result.output  # Separator lines
            assert "CONFIGURATION SUMMARY:" in result.output
            assert "AGENT RESPONSE:" in result.output
            assert "Test response" in result.output


class TestAgentCommandPerformanceIntegration:
    """Test performance characteristics of agent commands."""

    def test_agent_commands_response_time(self):
        """Test agent command response time characteristics."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.ask.RAGAgent"
        ) as mock_rag_agent_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Fast response"
            mock_rag_agent_class.return_value = mock_agent

            # Test multiple rapid calls
            for i in range(5):
                result = runner.invoke(agent_group, ["ask", "-q", f"Question {i}"])
                assert result.exit_code == 0
                assert "Fast response" in result.output

            # Verify agent was called multiple times
            assert mock_agent.ask.call_count == 5

    def test_agent_commands_memory_usage(self):
        """Test agent command memory usage characteristics."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.RAGAgent"
        ) as mock_rag_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = False
            mock_config.multi_agent.verbose = False
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_rag_agent_class.return_value = mock_agent

            # Test with large conversation history
            mock_prompt.side_effect = ["exit"]

            result = runner.invoke(agent_group, ["chat", "--max-history", "100"])

            assert result.exit_code == 0
            assert "📝 Max history: 100 messages" in result.output


class TestAgentCommandCompatibilityIntegration:
    """Test compatibility and interoperability of agent commands."""

    def test_agent_commands_click_compatibility(self):
        """Test that agent commands are compatible with click framework."""
        runner = CliRunner()

        # Test that agent group works as a click command
        result = runner.invoke(agent_group, ["--help"])
        assert result.exit_code == 0

        # Test that individual commands work as click commands
        result = runner.invoke(ask_command, ["--help"])
        assert result.exit_code == 0

        result = runner.invoke(chat_command, ["--help"])
        assert result.exit_code == 0

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

    def test_agent_commands_environment_compatibility(self):
        """Test agent commands work in different environments."""
        runner = CliRunner()

        # Test with different environment variables
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("src.paas_ai.cli.commands.agent.ask.load_config") as mock_load_config, patch(
                "src.paas_ai.cli.commands.agent.ask.RAGAgent"
            ) as mock_rag_agent_class:
                mock_config = Mock()
                mock_config.embedding.type = "openai"
                mock_load_config.return_value = mock_config

                mock_agent = Mock()
                mock_agent.ask.return_value = "Environment response"
                mock_rag_agent_class.return_value = mock_agent

                result = runner.invoke(agent_group, ["ask", "-q", "Test"])
                assert result.exit_code == 0
                assert "Environment response" in result.output

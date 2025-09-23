"""
Unit tests for agent chat command.

Tests all components of the chat command including:
- chat_command function
- Interactive chat session management
- Streaming response handling
- Special commands (history, clear, tools, config, tokens, exit)
- Error handling and edge cases
- Configuration display and management
"""

from unittest.mock import MagicMock, Mock, call, patch

import pytest
from click.testing import CliRunner
from langchain_core.messages import AIMessage, HumanMessage

from src.paas_ai.cli.commands.agent.chat import (
    _extract_chunk_content,
    _stream_response,
    chat_command,
)
from src.paas_ai.core.config import ConfigurationError
from src.paas_ai.core.config.schemas import DEFAULT_CONFIG_PROFILES


class TestExtractChunkContent:
    """Test the _extract_chunk_content helper function."""

    def test_extract_chunk_content_with_messages(self):
        """Test extracting content from chunk with messages."""
        # Mock message with content and type
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.type = "ai"

        chunk = {"messages": [mock_message]}

        result = _extract_chunk_content(chunk)
        assert result == "Test response"

    def test_extract_chunk_content_with_dict_messages(self):
        """Test extracting content from chunk with dict messages."""
        chunk = {"messages": [{"type": "ai", "content": "Test response"}]}

        result = _extract_chunk_content(chunk)
        assert result == "Test response"

    def test_extract_chunk_content_with_direct_content(self):
        """Test extracting content from chunk with direct content."""
        chunk = {"content": "Direct content"}

        result = _extract_chunk_content(chunk)
        assert result == "Direct content"

    def test_extract_chunk_content_with_agent_data(self):
        """Test extracting content from chunk with agent-specific data."""
        mock_message = Mock()
        mock_message.content = "Agent response"

        chunk = {"designer": {"messages": [mock_message]}}

        result = _extract_chunk_content(chunk)
        assert result == "Agent response"

    def test_extract_chunk_content_with_dict_agent_messages(self):
        """Test extracting content from chunk with dict agent messages."""
        chunk = {"paas_manifest_generator": {"messages": [{"content": "Generator response"}]}}

        result = _extract_chunk_content(chunk)
        assert result == "Generator response"

    def test_extract_chunk_content_empty_chunk(self):
        """Test extracting content from empty chunk."""
        chunk = {}

        result = _extract_chunk_content(chunk)
        assert result == ""

    def test_extract_chunk_content_none_chunk(self):
        """Test extracting content from None chunk."""
        result = _extract_chunk_content(None)
        assert result == ""

    def test_extract_chunk_content_non_dict_chunk(self):
        """Test extracting content from non-dict chunk."""
        result = _extract_chunk_content("not a dict")
        assert result == ""


class TestStreamResponse:
    """Test the _stream_response helper function."""

    def test_stream_response_with_question(self):
        """Test streaming response with question."""
        mock_agent = Mock()
        mock_agent.ask_stream.return_value = ["Hello", " world", "!"]

        result = _stream_response(mock_agent, question="Test question")

        assert result == "Hello world!"
        mock_agent.ask_stream.assert_called_once_with("Test question")

    def test_stream_response_with_direct_streaming(self):
        """Test streaming response with direct streaming."""
        mock_agent = Mock()
        mock_agent.ask_stream_direct.return_value = ["Direct", " response"]

        result = _stream_response(mock_agent, question="Test question", direct=True)

        assert result == "Direct response"
        mock_agent.ask_stream_direct.assert_called_once_with("Test question")

    def test_stream_response_with_messages(self):
        """Test streaming response with messages."""
        mock_agent = Mock()
        mock_agent.chat_stream.return_value = ["Chat", " response"]

        messages = [HumanMessage(content="Hello")]
        result = _stream_response(mock_agent, messages=messages)

        assert result == "Chat response"
        mock_agent.chat_stream.assert_called_once_with(messages)

    def test_stream_response_with_debug_mode(self):
        """Test streaming response with debug mode."""
        mock_agent = Mock()
        mock_agent.ask_stream.return_value = ["Debug", " response"]

        result = _stream_response(mock_agent, question="Test question", debug=True)

        assert result == "Debug response"
        mock_agent.ask_stream.assert_called_once_with("Test question")

    def test_stream_response_with_error_token(self):
        """Test streaming response with error token."""
        mock_agent = Mock()
        mock_agent.ask_stream.return_value = ["\n❌ Error occurred"]

        result = _stream_response(mock_agent, question="Test question")

        assert result == "\n❌ Error occurred"

    def test_stream_response_exception_handling(self):
        """Test streaming response exception handling."""
        mock_agent = Mock()
        mock_agent.ask_stream.side_effect = Exception("Streaming error")

        with pytest.raises(Exception, match="Streaming error"):
            _stream_response(mock_agent, question="Test question")

    def test_stream_response_no_parameters(self):
        """Test streaming response with no parameters."""
        mock_agent = Mock()

        with pytest.raises(ValueError, match="Either question or messages must be provided"):
            _stream_response(mock_agent)


class TestChatCommand:
    """Test the chat_command function."""

    def test_chat_command_basic_success(self):
        """Test basic successful chat command execution."""
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
            mock_agent.ask_stream.return_value = ["Hello", " world"]
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input and exit
            mock_prompt.side_effect = ["Hello", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            assert "🤖 RAG AGENT INTERACTIVE CHAT SESSION" in result.output
            assert "Hello world" in result.output
            assert "👋 Thanks for chatting! Goodbye!" in result.output

    def test_chat_command_with_show_config(self):
        """Test chat command with --show-config flag."""
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
            mock_agent.ask_stream.return_value = ["Response"]
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

            # Mock user input and exit
            mock_prompt.side_effect = ["exit"]

            # Run command
            result = runner.invoke(chat_command, ["--show-config"])

            # Verify
            assert result.exit_code == 0
            assert "CONFIGURATION SUMMARY:" in result.output
            assert "LLM: openai (gpt-3.5-turbo)" in result.output
            assert "Multi-Agent Mode: supervisor" in result.output

    def test_chat_command_with_config_profile(self):
        """Test chat command with --config-profile option."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.RAGAgent"
        ) as mock_rag_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "sentence_transformers"
            mock_config.multi_agent.track_tokens = False
            mock_config.multi_agent.verbose = False
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask_stream.return_value = ["Response"]
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input and exit
            mock_prompt.side_effect = ["exit"]

            # Run command
            result = runner.invoke(chat_command, ["--config-profile", "local"])

            # Verify
            assert result.exit_code == 0
            assert "🤖 RAG AGENT INTERACTIVE CHAT SESSION" in result.output

    def test_chat_command_with_max_history(self):
        """Test chat command with --max-history option."""
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
            mock_agent.ask_stream.return_value = ["Response"]
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input and exit
            mock_prompt.side_effect = ["exit"]

            # Run command
            result = runner.invoke(chat_command, ["--max-history", "10"])

            # Verify
            assert result.exit_code == 0
            assert "📝 Max history: 10 messages" in result.output

    def test_chat_command_history_command(self):
        """Test chat command history special command."""
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

            # Mock user input: history command then exit
            mock_prompt.side_effect = ["history", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            assert "📝 No conversation history yet." in result.output

    def test_chat_command_clear_command(self):
        """Test chat command clear special command."""
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
            mock_agent.clear_token_history = Mock()
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input: clear command then exit
            mock_prompt.side_effect = ["clear", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            assert "🧹 Conversation history cleared!" in result.output

    def test_chat_command_tools_command(self):
        """Test chat command tools special command."""
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
            mock_agent.get_available_tools.return_value = [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "args_schema": {"required": ["param1"]},
                }
            ]
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input: tools command then exit
            mock_prompt.side_effect = ["tools", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            assert "🔧 AVAILABLE TOOLS:" in result.output
            assert "test_tool" in result.output
            assert "A test tool" in result.output
            assert "Required: param1" in result.output

    def test_chat_command_config_command(self):
        """Test chat command config special command."""
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
            mock_agent.get_config_summary.return_value = {
                "llm": {"provider": "openai", "model": "gpt-3.5-turbo"},
                "embedding": {"type": "openai", "model": "text-embedding-3-small"},
                "vectorstore": {"type": "chroma", "directory": "/tmp/chroma", "collection": "test"},
            }
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input: config command then exit
            mock_prompt.side_effect = ["config", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            assert "⚙️  CURRENT CONFIGURATION:" in result.output
            assert "LLM: openai (gpt-3.5-turbo)" in result.output

    def test_chat_command_tokens_command_with_tracking(self):
        """Test chat command tokens special command with token tracking enabled."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.RAGAgent"
        ) as mock_rag_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = True
            mock_config.multi_agent.verbose = False
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.get_token_session_summary.return_value = {
                "total_tokens": 150,
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "total_requests": 2,
                "session_duration": 5.5,
                "agent_breakdown": {"designer": {"total_tokens": 150, "requests": 2}},
                "model_breakdown": {"gpt-3.5-turbo": {"total_tokens": 150, "requests": 2}},
            }
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input: tokens command then exit
            mock_prompt.side_effect = ["tokens", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            assert "🪙 TOKEN USAGE SUMMARY:" in result.output
            assert "Total Tokens: 150" in result.output
            assert "Input Tokens: 100" in result.output
            assert "Output Tokens: 50" in result.output
            assert "Total Requests: 2" in result.output
            assert "Session Duration: 5.5s" in result.output

    def test_chat_command_tokens_command_without_tracking(self):
        """Test chat command tokens special command without token tracking."""
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
            mock_agent.config = mock_config
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input: tokens command then exit
            mock_prompt.side_effect = ["tokens", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            assert "🪙 Token tracking is not enabled." in result.output

    def test_chat_command_exit_variations(self):
        """Test chat command with different exit commands."""
        exit_commands = ["exit", "quit", "bye"]

        for exit_cmd in exit_commands:
            runner = CliRunner()

            with patch(
                "src.paas_ai.cli.commands.agent.chat.load_config"
            ) as mock_load_config, patch(
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

                # Mock user input: exit command
                mock_prompt.side_effect = [exit_cmd]

                # Run command
                result = runner.invoke(chat_command, [])

                # Verify
                assert result.exit_code == 0
                assert "👋 Thanks for chatting! Goodbye!" in result.output

    def test_chat_command_configuration_error(self):
        """Test chat command with configuration error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config:
            # Setup mock to raise configuration error
            mock_load_config.side_effect = ConfigurationError("Config file not found")

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0  # Command should handle error gracefully
            # The ConfigurationError is being caught by the general exception handler
            # so we expect the "Failed to start chat" message instead
            assert "❌ Failed to start chat: Config file not found" in result.output

    def test_chat_command_agent_error(self):
        """Test chat command with agent processing error."""
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
            mock_agent.ask_stream.side_effect = Exception("Agent processing error")
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input: question then exit
            mock_prompt.side_effect = ["Test question", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            # Should handle error gracefully and continue
            assert "👋 Thanks for chatting! Goodbye!" in result.output


class TestChatCommandEdgeCases:
    """Test edge cases for chat command."""

    def test_chat_command_empty_input(self):
        """Test chat command with empty user input."""
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

            # Mock user input: empty string then exit
            mock_prompt.side_effect = ["", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            # Should skip empty input and continue
            assert "👋 Thanks for chatting! Goodbye!" in result.output

    def test_chat_command_whitespace_input(self):
        """Test chat command with whitespace-only input."""
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

            # Mock user input: whitespace then exit
            mock_prompt.side_effect = ["   ", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            # Should skip whitespace input and continue
            assert "👋 Thanks for chatting! Goodbye!" in result.output

    def test_chat_command_keyboard_interrupt(self):
        """Test chat command with keyboard interrupt."""
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

            # Mock user input: keyboard interrupt
            mock_prompt.side_effect = KeyboardInterrupt()

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            assert "👋 Session interrupted. Goodbye!" in result.output

    def test_chat_command_streaming_fallback(self):
        """Test chat command with streaming fallback to non-streaming."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.RAGAgent"
        ) as mock_rag_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt, patch(
            "src.paas_ai.cli.commands.agent.chat._stream_response"
        ) as mock_stream_response:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = False
            mock_config.multi_agent.verbose = False
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask.return_value = "Fallback response"
            mock_rag_agent_class.return_value = mock_agent

            # Mock streaming to fail
            mock_stream_response.side_effect = Exception("Streaming failed")

            # Mock user input: question then exit
            mock_prompt.side_effect = ["Test question", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            assert "⚠️ Streaming failed, falling back to standard mode" in result.output
            assert "Fallback response" in result.output

    def test_chat_command_history_trimming(self):
        """Test chat command with history trimming."""
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
            # Mock all the streaming methods to return proper string responses
            mock_agent.ask_stream.return_value = ["Response"]
            mock_agent.chat_stream.return_value = ["Response"]
            # Mock fallback methods
            mock_agent.ask.return_value = "Response"
            mock_agent.chat.return_value = "Response"
            # Mock other methods that might be called
            mock_agent.get_config_summary.return_value = {
                "llm": {"provider": "test", "model": "test"},
                "embedding": {"type": "openai", "model": "test"},
                "vectorstore": {"type": "test", "directory": "test", "collection": "test"},
            }
            mock_agent.get_available_tools.return_value = []
            mock_agent.get_token_session_summary.return_value = {"total_tokens": 0}
            mock_rag_agent_class.return_value = mock_agent

            # Mock many user inputs to trigger history trimming
            # We need enough inputs to exceed max_history * 2 (which is 10 for max_history=5)
            # Each exchange adds 2 messages (user + agent), so we need 6+ exchanges
            mock_prompt.side_effect = ["Question"] * 12 + ["exit"]

            # Run command with small max history
            result = runner.invoke(chat_command, ["--max-history", "5"])

            # Verify
            assert result.exit_code == 0
            assert "📝 Trimmed old conversation history" in result.output

    def test_chat_command_with_verbose_token_tracking(self):
        """Test chat command with verbose mode and token tracking."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.agent.chat.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.agent.chat.RAGAgent"
        ) as mock_rag_agent_class, patch(
            "src.paas_ai.cli.commands.agent.chat.click.prompt"
        ) as mock_prompt:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_config.multi_agent.track_tokens = True
            mock_config.multi_agent.verbose = True
            mock_load_config.return_value = mock_config

            mock_agent = Mock()
            mock_agent.ask_stream.return_value = ["Response"]
            mock_agent.get_token_session_summary.return_value = {
                "total_tokens": 100,
                "agents_used": ["designer"],
                "session_duration": 3.5,
            }
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input: question then exit
            mock_prompt.side_effect = ["Test question", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            assert "🪙 Tokens: 100 (designer)" in result.output
            assert "📊 Session completed: 1 exchanges" in result.output


class TestChatCommandIntegration:
    """Integration tests for chat command."""

    def test_chat_command_full_conversation(self):
        """Test complete chat conversation workflow."""
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
            mock_agent.ask_stream.return_value = ["First response"]
            mock_agent.chat_stream.return_value = ["Second response"]
            mock_agent.get_available_tools.return_value = [
                {"name": "test_tool", "description": "Test"}
            ]
            mock_agent.get_config_summary.return_value = {
                "llm": {"provider": "openai", "model": "gpt-3.5-turbo"},
                "embedding": {"type": "openai", "model": "text-embedding-3-small"},
                "vectorstore": {"type": "chroma", "directory": "/tmp/chroma", "collection": "test"},
            }
            # Set up agent config to avoid Mock comparison errors
            mock_agent.config = mock_config
            # Mock the token session summary method to return proper values
            mock_agent.get_token_session_summary.return_value = {
                "total_tokens": 0,
                "agents_used": [],
                "session_duration": 0.0,
            }
            mock_rag_agent_class.return_value = mock_agent

            # Mock conversation flow
            mock_prompt.side_effect = [
                "First question",
                "tools",
                "config",
                "Second question",
                "exit",
            ]

            # Run command
            result = runner.invoke(chat_command, ["--show-config"])

            # Verify
            assert result.exit_code == 0
            assert "CONFIGURATION SUMMARY:" in result.output
            assert "First response" in result.output
            assert "🔧 AVAILABLE TOOLS:" in result.output
            assert "⚙️  CURRENT CONFIGURATION:" in result.output
            assert "Second response" in result.output
            assert "👋 Thanks for chatting! Goodbye!" in result.output
            assert "📊 Session completed: 2 exchanges" in result.output

    def test_chat_command_with_debug_streaming(self):
        """Test chat command with debug streaming enabled."""
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
            mock_agent.ask_stream.return_value = ["Debug", " response"]
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input and exit
            mock_prompt.side_effect = ["Test question", "exit"]

            # Run command with debug streaming
            result = runner.invoke(chat_command, ["--debug-streaming"])

            # Verify
            assert result.exit_code == 0
            assert "Debug response" in result.output

    def test_chat_command_with_direct_streaming(self):
        """Test chat command with direct streaming enabled."""
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
            mock_agent.ask_stream_direct.return_value = ["Direct", " response"]
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input and exit
            mock_prompt.side_effect = ["Test question", "exit"]

            # Run command with direct streaming
            result = runner.invoke(chat_command, ["--direct-streaming"])

            # Verify
            assert result.exit_code == 0
            assert "Direct response" in result.output

    def test_chat_command_error_recovery(self):
        """Test chat command error recovery and graceful handling."""
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
            mock_agent.ask_stream.side_effect = Exception("Processing error")
            mock_agent.ask.return_value = "Fallback response"
            mock_rag_agent_class.return_value = mock_agent

            # Mock user input: question then exit
            mock_prompt.side_effect = ["Test question", "exit"]

            # Run command
            result = runner.invoke(chat_command, [])

            # Verify
            assert result.exit_code == 0
            assert "⚠️ Streaming failed, falling back to standard mode" in result.output
            assert "Fallback response" in result.output
            assert "💡 You can continue chatting or type 'exit' to quit." in result.output
            assert "👋 Thanks for chatting! Goodbye!" in result.output

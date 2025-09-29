"""
Agent chat command - Interactive chat session with the RAG agent.
"""

from typing import Optional

import click

from paas_ai.core.agents.multi_agent_system import MultiAgentSystem
from paas_ai.core.agents.persistence import generate_thread_id
from paas_ai.core.config import ConfigurationError, load_config
from paas_ai.utils.logging import get_logger

logger = get_logger("paas_ai.cli.agent.chat")


def _extract_chunk_content(chunk) -> str:
    """
    Extract displayable content from a streaming chunk.

    Args:
        chunk: Streaming chunk from the coordination system

    Returns:
        str: Content to display, or empty string if no displayable content
    """
    # Debug: print the chunk structure
    logger.debug(f"Processing chunk: {chunk}")

    # Handle different chunk formats from LangGraph
    if isinstance(chunk, dict):
        # Check for message content in various formats
        if "messages" in chunk:
            messages = chunk["messages"]
            if messages and isinstance(messages, list):
                # Get the last message if it's an AI message
                last_msg = messages[-1]
                if hasattr(last_msg, "content") and hasattr(last_msg, "type"):
                    if last_msg.type == "ai" or getattr(last_msg, "role", None) == "assistant":
                        return last_msg.content
                elif isinstance(last_msg, dict):
                    if last_msg.get("type") == "ai" or last_msg.get("role") == "assistant":
                        return last_msg.get("content", "")

        # Check for direct content
        if "content" in chunk:
            return chunk["content"]

        # Check for agent-specific content
        for agent_name in ["designer", "paas_manifest_generator", "supervisor"]:
            if agent_name in chunk:
                agent_data = chunk[agent_name]
                if isinstance(agent_data, dict) and "messages" in agent_data:
                    messages = agent_data["messages"]
                    if messages and isinstance(messages, list):
                        last_msg = messages[-1]
                        if hasattr(last_msg, "content"):
                            return last_msg.content
                        elif isinstance(last_msg, dict) and "content" in last_msg:
                            return last_msg["content"]

    return ""


def _stream_response(
    agent, question=None, messages=None, debug=False, direct=False, thread_id=None
):
    """
    Stream response from agent and return the complete response.

    Args:
        agent: The agent instance
        question: Question for ask_stream (if not using chat)
        messages: Messages for chat_stream (if not using ask)
        debug: If True, show detailed debugging info
        direct: If True, use direct streaming (bypass coordinator)
        thread_id: Thread ID for conversation persistence

    Returns:
        str: Complete response text
    """
    response_parts = []

    try:
        # Choose streaming method based on parameters
        if question is not None:
            if direct:
                stream = agent.ask_stream_direct(question)
            else:
                stream = agent.ask_stream(question)
        elif messages is not None:
            # For chat, use streaming with thread persistence
            stream = agent.chat_stream(messages, thread_id=thread_id)
        else:
            raise ValueError("Either question or messages must be provided")

        token_count = 0
        for token in stream:
            token_count += 1

            if debug:
                # Debug: show token info
                click.echo(
                    click.style(f"\n[DEBUG] Token {token_count}: '{token}'", fg="cyan", dim=True)
                )

            # Check for error tokens
            if token.startswith("\n❌"):
                click.echo(click.style(token, fg="red"))
                response_parts.append(token)
                break

            # Display the token immediately (real streaming!)
            if not debug:
                click.echo(token, nl=False)
            response_parts.append(token)

        if debug:
            click.echo(
                click.style(f"\n[DEBUG] Total tokens streamed: {token_count}", fg="cyan", dim=True)
            )
            click.echo(
                click.style(
                    f"\n[DEBUG] Final response: {''.join(response_parts)}", fg="green", dim=True
                )
            )

        return "".join(response_parts)

    except Exception as e:
        # Return error for fallback handling
        raise e


@click.command()
@click.option("--config-profile", help="Override config profile for this operation")
@click.option("--show-config", is_flag=True, help="Show configuration summary")
@click.option("--thread-id", help="Conversation thread ID (auto-generated if not provided)")
@click.option(
    "--debug-streaming", is_flag=True, help="Debug streaming chunks (shows raw chunk data)"
)
@click.option(
    "--direct-streaming", is_flag=True, help="Use direct agent streaming (bypass coordinator)"
)
def chat_command(
    config_profile: Optional[str],
    show_config: bool,
    thread_id: Optional[str],
    debug_streaming: bool,
    direct_streaming: bool,
):
    """
    Start an interactive chat session with persistent conversation history.

    Uses LangGraph's built-in persistence to maintain conversation context
    across sessions. Each conversation is identified by a unique thread ID.

    Examples:

        # Start a basic chat session (auto-generates thread ID)
        paas-ai agent chat

        # Continue an existing conversation
        paas-ai agent chat --thread-id chat_1234567890_abcd1234

        # Start with configuration display
        paas-ai agent chat --show-config

    Available commands during chat:

        • Ask any question about your knowledge base
        • 'tools' - Show available agent tools
        • 'config' - Show current configuration
        • 'tokens' - Show token usage summary (when tracking enabled)
        • 'exit', 'quit', or 'bye' - End session
    """
    try:
        from langchain_core.messages import AIMessage, HumanMessage

        # Load configuration with profile override
        if config_profile:
            # Use the config profiles system like RAG commands
            from ....core.config.schemas import DEFAULT_CONFIG_PROFILES

            if config_profile in DEFAULT_CONFIG_PROFILES:
                config = DEFAULT_CONFIG_PROFILES[config_profile]
                logger.info(f"Using config profile: {config_profile}")
            else:
                logger.warning(f"Unknown config profile '{config_profile}', using default")
                config = load_config()
        else:
            config = load_config()

        logger.info(f"Using configuration with {config.embedding.type} embeddings")

        # Initialize agent system
        agent = MultiAgentSystem(config)

        # Generate thread ID if not provided
        if not thread_id:
            thread_id = generate_thread_id()
            click.echo(f"📝 Starting new conversation (Thread: {thread_id})")
        else:
            click.echo(f"📝 Continuing conversation (Thread: {thread_id})")

        # Show config summary if requested
        if show_config:
            config_summary = agent.get_config_summary()
            click.echo("\n" + "=" * 60)
            click.echo("CONFIGURATION SUMMARY:")
            click.echo("=" * 60)
            click.echo(
                f"LLM: {config_summary['llm']['provider']} ({config_summary['llm']['model']})"
            )
            click.echo(
                f"Embedding: {config_summary['embedding']['type']} ({config_summary['embedding']['model']})"
            )
            click.echo(
                f"VectorStore: {config_summary['vectorstore']['type']} -> {config_summary['vectorstore']['directory']}"
            )
            click.echo(f"Collection: {config_summary['vectorstore']['collection']}")

            ma_config = config_summary["multi_agent"]
            click.echo(f"Multi-Agent Mode: {ma_config['mode']}")
            click.echo(f"Agents: {', '.join(ma_config['agents'])}")
            click.echo(f"Token Tracking: {'ON' if ma_config['track_tokens'] else 'OFF'}")
            click.echo(f"Verbose Mode: {'ON' if ma_config['verbose'] else 'OFF'}")

            click.echo("=" * 60 + "\n")

        # Conversation history is now managed by LangGraph persistence

        click.echo("\n" + "=" * 60)
        click.echo("🤖 MULTI-AGENT INTERACTIVE CHAT SESSION")
        click.echo("=" * 60)
        click.echo("💡 Commands:")
        click.echo("  • 'exit' or 'quit' - End the session")
        click.echo("  • 'tools' - Show available tools")
        click.echo("  • 'config' - Show current configuration")
        click.echo("  • 'tokens' - Show token usage summary")
        click.echo("=" * 60)
        click.echo(f"🧵 Thread: {thread_id}")
        click.echo("=" * 60 + "\n")

        session_count = 0
        total_exchanges = 0

        while True:
            try:
                # Get user input
                question = click.prompt(click.style("You", fg="blue", bold=True), type=str)

                # Handle special commands
                if question.lower() in ["exit", "quit", "bye"]:
                    click.echo(click.style("\n👋 Thanks for chatting! Goodbye!", fg="green"))
                    break

                # History and clearing are handled by LangGraph persistence

                if question.lower() == "tools":
                    tools = agent.get_available_tools()
                    click.echo(click.style("\n🔧 AVAILABLE TOOLS:", fg="cyan", bold=True))
                    click.echo("=" * 40)
                    for tool in tools:
                        click.echo(f"• {click.style(tool['name'], fg='magenta', bold=True)}")
                        click.echo(f"  {tool['description'].strip()}")
                        if tool.get("args_schema"):
                            required_args = tool["args_schema"].get("required", [])
                            if required_args:
                                click.echo(f"  Required: {', '.join(required_args)}")
                        click.echo()
                    continue

                if question.lower() == "config":
                    config_summary = agent.get_config_summary()
                    click.echo(click.style("\n⚙️  CURRENT CONFIGURATION:", fg="cyan", bold=True))
                    click.echo("=" * 40)
                    click.echo(
                        f"LLM: {config_summary['llm']['provider']} ({config_summary['llm']['model']})"
                    )
                    click.echo(
                        f"Embedding: {config_summary['embedding']['type']} ({config_summary['embedding']['model']})"
                    )
                    click.echo(
                        f"VectorStore: {config_summary['vectorstore']['type']} -> {config_summary['vectorstore']['directory']}"
                    )
                    click.echo(f"Collection: {config_summary['vectorstore']['collection']}")
                    click.echo("=" * 40 + "\n")
                    continue

                if question.lower() == "tokens":
                    if hasattr(agent, "config") and agent.config.multi_agent.track_tokens:
                        token_summary = agent.get_token_session_summary()
                        click.echo(click.style("\n🪙 TOKEN USAGE SUMMARY:", fg="cyan", bold=True))
                        click.echo("=" * 40)

                        if token_summary.get("total_tokens", 0) > 0:
                            click.echo(f"Total Tokens: {token_summary['total_tokens']}")
                            click.echo(f"Input Tokens: {token_summary['total_input_tokens']}")
                            click.echo(f"Output Tokens: {token_summary['total_output_tokens']}")
                            click.echo(f"Total Requests: {token_summary['total_requests']}")
                            click.echo(
                                f"Session Duration: {token_summary['session_duration']:.1f}s"
                            )

                            if token_summary.get("agent_breakdown"):
                                click.echo("\nPer-Agent Breakdown:")
                                for agent_name, stats in token_summary["agent_breakdown"].items():
                                    click.echo(
                                        f"  • {agent_name}: {stats['total_tokens']} tokens ({stats['requests']} requests)"
                                    )

                            if token_summary.get("model_breakdown"):
                                click.echo("\nPer-Model Breakdown:")
                                for model_name, stats in token_summary["model_breakdown"].items():
                                    click.echo(
                                        f"  • {model_name}: {stats['total_tokens']} tokens ({stats['requests']} requests)"
                                    )
                        else:
                            click.echo("No token usage recorded yet.")

                        click.echo("=" * 40 + "\n")
                    else:
                        click.echo(click.style("🪙 Token tracking is not enabled.\n", fg="yellow"))
                    continue

                # Skip empty questions
                if not question.strip():
                    continue

                # Create user message (LangGraph will handle persistence)
                user_message = HumanMessage(content=question)

                # Get agent response using conversation history
                click.echo(click.style("🤔 Agent is thinking...", fg="yellow"))

                # Start response display
                # TODO: Show this when the response actually starts
                click.echo(f"\n{click.style('🤖 Agent:', fg='green', bold=True)} ", nl=False)

                # Stream the response
                try:
                    # Use chat with single message - LangGraph will load conversation history automatically
                    response = _stream_response(
                        agent,
                        messages=[user_message],
                        debug=debug_streaming,
                        direct=direct_streaming,
                        thread_id=thread_id,
                    )

                    click.echo("\n")  # Add newline after streaming

                except Exception as e:
                    # Fallback to non-streaming if streaming fails
                    click.echo(
                        click.style(
                            f"\n⚠️ Streaming failed, falling back to standard mode: {e}",
                            fg="yellow",
                        )
                    )

                    # Fallback to non-streaming with thread persistence
                    response = agent.chat([user_message], thread_id=thread_id)
                    click.echo(f"{response}\n")

                # LangGraph automatically manages conversation history via checkpointer

                # Display response (response already shown during streaming)
                session_count += 1
                total_exchanges += 1

                # Get current configuration to check if we should show token info
                if (
                    hasattr(agent, "config")
                    and agent.config.multi_agent.verbose
                    and agent.config.multi_agent.track_tokens
                ):
                    # Get token session summary
                    token_summary = agent.get_token_session_summary()

                    # Format the session summary with token information
                    total_tokens = token_summary.get("total_tokens", 0)
                    agents_used = token_summary.get("agents_used", [])

                    if total_tokens > 0:
                        click.echo(
                            click.style(
                                f"💬 Exchanges: {session_count} | "
                                f"🪙 Tokens: {total_tokens} ({', '.join(agents_used)})",
                                fg="cyan",
                                dim=True,
                            )
                        )
                    else:
                        click.echo(
                            click.style(
                                f"💬 Exchanges in session: {session_count}",
                                fg="cyan",
                                dim=True,
                            )
                        )
                else:
                    click.echo(
                        click.style(
                            f"💬 Exchanges in session: {session_count}",
                            fg="cyan",
                            dim=True,
                        )
                    )

                click.echo()

            except KeyboardInterrupt:
                click.echo(click.style("\n\n👋 Session interrupted. Goodbye!", fg="yellow"))
                break
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                click.echo(click.style(f"❌ Error: {e}", fg="red"))
                click.echo(
                    click.style(
                        "💡 You can continue chatting or type 'exit' to quit.\n", fg="yellow"
                    )
                )

        # Session summary
        if total_exchanges > 0:
            if (
                hasattr(agent, "config")
                and agent.config.multi_agent.verbose
                and agent.config.multi_agent.track_tokens
            ):
                # Get final token session summary
                token_summary = agent.get_token_session_summary()
                total_tokens = token_summary.get("total_tokens", 0)
                agents_used = token_summary.get("agents_used", [])
                session_duration = token_summary.get("session_duration", 0)

                if total_tokens > 0:
                    click.echo(
                        click.style(
                            f"📊 Session completed: {total_exchanges} exchanges, "
                            f"🪙 {total_tokens} tokens used across {len(agents_used)} agents ({session_duration:.1f}s)",
                            fg="green",
                        )
                    )
                else:
                    click.echo(
                        click.style(
                            f"📊 Session completed: {total_exchanges} exchanges",
                            fg="green",
                        )
                    )
            else:
                click.echo(
                    click.style(
                        f"📊 Session completed: {total_exchanges} exchanges",
                        fg="green",
                    )
                )

        return True

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Try creating a config file with: paas-ai config init")
        click.echo(click.style(f"❌ Configuration error: {e}", fg="red"))
        click.echo(
            click.style("💡 Try creating a config file with: paas-ai config init", fg="yellow")
        )
        return False
    except Exception as e:
        logger.error(f"Failed to start chat: {e}")
        click.echo(click.style(f"❌ Failed to start chat: {e}", fg="red"))
        return False

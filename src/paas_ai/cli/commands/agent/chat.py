"""
Agent chat command - Interactive chat session with the RAG agent.
"""

import click
from typing import Optional

from ....core.config import load_config, ConfigurationError
from ....core.agents import RAGAgent
from ....utils.logging import get_logger

logger = get_logger("paas_ai.cli.agent.chat")


@click.command()
@click.option("--config-profile", help="Override config profile for this operation")
@click.option("--show-config", is_flag=True, help="Show configuration summary")
@click.option("--max-history", default=20, help="Maximum number of messages to keep in history")
def chat_command(config_profile: Optional[str], show_config: bool, max_history: int):
    """
    Start an interactive chat session with the RAG agent.
    
    This creates a persistent conversation where the agent remembers context
    from previous messages in the session. Use this for complex discussions
    or when you need to refer back to earlier parts of the conversation.
    
    Examples:
    
        # Start a basic chat session
        paas-ai agent chat
        
        # Start with configuration display
        paas-ai agent chat --show-config
        
        # Limit conversation history to 10 exchanges
        paas-ai agent chat --max-history 10
    
    Available commands during chat:
    
        • Ask any question about your knowledge base
        • 'history' - View conversation history  
        • 'clear' - Clear conversation history
        • 'tools' - Show available agent tools
        • 'config' - Show current configuration
        • 'exit', 'quit', or 'bye' - End session
    """
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Load configuration
        if config_profile:
            logger.warning(f"Config profile override not yet implemented: {config_profile}")
        
        config = load_config()
        logger.info(f"Using configuration with {config.embedding.type} embeddings")
        
        # Initialize agent
        agent = RAGAgent(config)
        
        # Show config summary if requested
        if show_config:
            config_summary = agent.get_config_summary()
            click.echo("\n" + "="*60)
            click.echo("CONFIGURATION SUMMARY:")
            click.echo("="*60)
            click.echo(f"LLM: {config_summary['llm']['provider']} ({config_summary['llm']['model']})")
            click.echo(f"Embedding: {config_summary['embedding']['type']} ({config_summary['embedding']['model']})")
            click.echo(f"VectorStore: {config_summary['vectorstore']['type']} -> {config_summary['vectorstore']['directory']}")
            click.echo(f"Collection: {config_summary['vectorstore']['collection']}")
            click.echo("="*60 + "\n")
        
        # Initialize conversation history
        conversation_history = []
        
        click.echo("\n" + "="*60)
        click.echo("🤖 RAG AGENT INTERACTIVE CHAT SESSION")
        click.echo("="*60)
        click.echo("💡 Commands:")
        click.echo("  • 'exit' or 'quit' - End the session")
        click.echo("  • 'clear' - Clear conversation history")
        click.echo("  • 'history' - Show conversation history")
        click.echo("  • 'tools' - Show available tools")
        click.echo("  • 'config' - Show current configuration")
        click.echo("="*60)
        click.echo(f"📝 Max history: {max_history} messages")
        click.echo("="*60 + "\n")
        
        session_count = 0
        
        while True:
            try:
                # Get user input
                question = click.prompt(click.style("You", fg="blue", bold=True), type=str)
                
                # Handle special commands
                if question.lower() in ['exit', 'quit', 'bye']:
                    click.echo(click.style("\n👋 Thanks for chatting! Goodbye!", fg="green"))
                    break
                
                if question.lower() == 'clear':
                    conversation_history = []
                    session_count = 0
                    click.echo(click.style("🧹 Conversation history cleared!\n", fg="yellow"))
                    continue
                
                if question.lower() == 'history':
                    if not conversation_history:
                        click.echo(click.style("📝 No conversation history yet.\n", fg="yellow"))
                    else:
                        click.echo(click.style("\n📜 CONVERSATION HISTORY:", fg="cyan", bold=True))
                        click.echo("="*50)
                        for i, msg in enumerate(conversation_history, 1):
                            role = "You" if isinstance(msg, HumanMessage) else "Agent"
                            color = "blue" if isinstance(msg, HumanMessage) else "green"
                            click.echo(f"{click.style(f'{i}. {role}:', fg=color, bold=True)} {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
                        click.echo("="*50 + "\n")
                    continue
                
                if question.lower() == 'tools':
                    tools = agent.get_available_tools()
                    click.echo(click.style("\n🔧 AVAILABLE TOOLS:", fg="cyan", bold=True))
                    click.echo("="*40)
                    for tool in tools:
                        click.echo(f"• {click.style(tool['name'], fg='magenta', bold=True)}")
                        click.echo(f"  {tool['description'].strip()}")
                        if tool.get('args_schema'):
                            required_args = tool['args_schema'].get('required', [])
                            if required_args:
                                click.echo(f"  Required: {', '.join(required_args)}")
                        click.echo()
                    continue
                
                if question.lower() == 'config':
                    config_summary = agent.get_config_summary()
                    click.echo(click.style("\n⚙️  CURRENT CONFIGURATION:", fg="cyan", bold=True))
                    click.echo("="*40)
                    click.echo(f"LLM: {config_summary['llm']['provider']} ({config_summary['llm']['model']})")
                    click.echo(f"Embedding: {config_summary['embedding']['type']} ({config_summary['embedding']['model']})")
                    click.echo(f"VectorStore: {config_summary['vectorstore']['type']} -> {config_summary['vectorstore']['directory']}")
                    click.echo(f"Collection: {config_summary['vectorstore']['collection']}")
                    click.echo("="*40 + "\n")
                    continue
                
                # Skip empty questions
                if not question.strip():
                    continue
                
                # Add user message to history
                user_message = HumanMessage(content=question)
                conversation_history.append(user_message)
                
                # Get agent response using conversation history
                click.echo(click.style("🤔 Agent is thinking...", fg="yellow"))
                
                if len(conversation_history) == 1:
                    # First message, use simple ask
                    response = agent.ask(question)
                else:
                    # Use chat with conversation history
                    response = agent.chat(conversation_history)
                
                # Add agent response to history
                agent_message = AIMessage(content=response)
                conversation_history.append(agent_message)
                
                # Trim history if it gets too long
                if len(conversation_history) > max_history * 2:  # *2 because each exchange is 2 messages
                    conversation_history = conversation_history[-max_history * 2:]
                    click.echo(click.style("📝 Trimmed old conversation history", fg="yellow", dim=True))
                
                # Display response
                session_count += 1
                click.echo(f"\n{click.style('🤖 Agent:', fg='green', bold=True)} {response}\n")
                click.echo(click.style(f"💬 Messages in session: {len(conversation_history)} | Exchanges: {session_count}", fg="cyan", dim=True))
                click.echo()
                
            except KeyboardInterrupt:
                click.echo(click.style("\n\n👋 Session interrupted. Goodbye!", fg="yellow"))
                break
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                click.echo(click.style(f"❌ Error: {e}", fg="red"))
                click.echo(click.style("💡 You can continue chatting or type 'exit' to quit.\n", fg="yellow"))
        
        # Session summary
        if session_count > 0:
            click.echo(click.style(f"📊 Session completed: {session_count} exchanges, {len(conversation_history)} total messages", fg="green"))
        
        return True
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Try creating a config file with: paas-ai config init")
        click.echo(click.style(f"❌ Configuration error: {e}", fg="red"))
        click.echo(click.style("💡 Try creating a config file with: paas-ai config init", fg="yellow"))
        return False
    except Exception as e:
        logger.error(f"Failed to start chat: {e}")
        click.echo(click.style(f"❌ Failed to start chat: {e}", fg="red"))
        return False 